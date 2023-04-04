#include "precomp.h"

#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

using namespace std;
using namespace paraformer;

ModelImp::ModelImp(const char* path,int nNumThread, bool quantize)
{
    string model_path;
    string cmvn_path;
    string config_path;

    if(quantize)
    {
        model_path = pathAppend(path, "model_quant.onnx");
    }else{
        model_path = pathAppend(path, "model.onnx");
    }
    cmvn_path = pathAppend(path, "am.mvn");
    config_path = pathAppend(path, "config.yaml");

    fe = new FeatureExtract(3);

    //sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(nNumThread);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    wstring wstrPath = strToWstr(model_path);
    m_session = new Ort::Session(env, wstrPath.c_str(), sessionOptions);
#else
    m_session = new Ort::Session(env, model_path.c_str(), sessionOptions);
#endif

    string strName;
    getInputName(m_session, strName);
    m_strInputNames.push_back(strName.c_str());
    getInputName(m_session, strName,1);
    m_strInputNames.push_back(strName);
    
    getOutputName(m_session, strName);
    m_strOutputNames.push_back(strName);
    getOutputName(m_session, strName,1);
    m_strOutputNames.push_back(strName);

    for (auto& item : m_strInputNames)
        m_szInputNames.push_back(item.c_str());
    for (auto& item : m_strOutputNames)
        m_szOutputNames.push_back(item.c_str());
    vocab = new Vocab(config_path.c_str());
    load_cmvn(cmvn_path.c_str());
}

ModelImp::~ModelImp()
{
    long total = total_fe_time+total_lfr_time+total_cmvn_time+total_fwd_time;
    printf("total_fe_time %ld, %05lf\n",   total_fe_time, (float)total_fe_time/total);
    printf("total_lfr_time %ld, %05lf\n",  total_lfr_time, (float)total_lfr_time/total);
    printf("total_cmvn_time %ld, %05lf\n", total_cmvn_time, (float)total_cmvn_time/total);
    printf("total_fwd_time %ld, %05lf\n",  total_fwd_time, (float)total_fwd_time/total);

    if(fe)
        delete fe;
    if (m_session)
    {
        delete m_session;
        m_session = nullptr;
    }
    if(vocab)
        delete vocab;
}

void ModelImp::reset()
{
    fe->reset();
}

void ModelImp::apply_lfr(Tensor<float>*& din)
{
    int mm = din->size[2];
    int ll = ceil(mm / 6.0);
    Tensor<float>* tmp = new Tensor<float>(ll, 560);
    int out_offset = 0;
    for (int i = 0; i < ll; i++) {
        for (int j = 0; j < 7; j++) {
            int idx = i * 6 + j - 3;
            if (idx < 0) {
                idx = 0;
            }
            if (idx >= mm) {
                idx = mm - 1;
            }
            memcpy(tmp->buff + out_offset, din->buff + idx * 80,
                sizeof(float) * 80);
            out_offset += 80;
        }
    }
    delete din;
    din = tmp;
}

void ModelImp::load_cmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    string line;

    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> line_item{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (line_item[0] == "<AddShift>") {
            getline(cmvn_stream, line);
            istringstream means_lines_stream(line);
            vector<string> means_lines{istream_iterator<string>{means_lines_stream}, istream_iterator<string>{}};
            if (means_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < means_lines.size() - 1; j++) {
                    means_list.push_back(stof(means_lines[j]));
                }
                continue;
            }
        }
        else if (line_item[0] == "<Rescale>") {
            getline(cmvn_stream, line);
            istringstream vars_lines_stream(line);
            vector<string> vars_lines{istream_iterator<string>{vars_lines_stream}, istream_iterator<string>{}};
            if (vars_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < vars_lines.size() - 1; j++) {
                    vars_list.push_back(stof(vars_lines[j])*scale);
                }
                continue;
            }
        }
    }
}

void ModelImp::apply_cmvn(Tensor<float>* din)
{
    const float* var;
    const float* mean;
    var = vars_list.data();
    mean= means_list.data();

    int m = din->size[2];
    int n = din->size[3];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            din->buff[idx] = (din->buff[idx] + mean[j]) * var[j];
        }
    }
}

string ModelImp::greedy_search(float * in, int nLen )
{
    vector<int> hyps;
    int Tmax = nLen;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        findmax(in + i * 8404, 8404, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab->vector2stringV2(hyps);
}

string ModelImp::forward(float* din, int len, int flag)
{
    struct timeval fe_start, fe_end;
    struct timeval lfr_start, lfr_end;
    struct timeval cmvn_start, cmvn_end;
    struct timeval fwd_start, fwd_end;

    Tensor<float>* in;
    //extract feature
    gettimeofday(&fe_start, NULL);
    fe->insert(din, len, flag);
    fe->fetch(in);
    gettimeofday(&fe_end, NULL);
    long seconds = (fe_end.tv_sec - fe_start.tv_sec);
    long taking_micros = ((seconds * 1000000) + fe_end.tv_usec) - (fe_start.tv_usec);
    total_fe_time += taking_micros;
    
    gettimeofday(&lfr_start, NULL);
    apply_lfr(in);
    gettimeofday(&lfr_end, NULL);
    seconds = (lfr_end.tv_sec - lfr_start.tv_sec);
    taking_micros = ((seconds * 1000000) + lfr_end.tv_usec) - (lfr_start.tv_usec);
    total_lfr_time += taking_micros;

    gettimeofday(&cmvn_start, NULL);
    apply_cmvn(in);
    gettimeofday(&cmvn_end, NULL);
    seconds = (cmvn_end.tv_sec - cmvn_start.tv_sec);
    taking_micros = ((seconds * 1000000) + cmvn_end.tv_usec) - (cmvn_start.tv_usec);
    total_cmvn_time += taking_micros;

    gettimeofday(&fwd_start, NULL);
    Ort::RunOptions run_option;
    std::array<int64_t, 3> input_shape_{ in->size[0],in->size[2],in->size[3] };
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(m_memoryInfo,
        in->buff,
        in->buff_size,
        input_shape_.data(),
        input_shape_.size());

    std::vector<int32_t> feats_len{ in->size[2] };
    std::vector<int64_t> feats_len_dim{ 1 };
    Ort::Value onnx_feats_len = Ort::Value::CreateTensor(
        m_memoryInfo,
        feats_len.data(),
        feats_len.size() * sizeof(int32_t),
        feats_len_dim.data(),
        feats_len_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_feats_len));

    string result;
    try {

        auto outputTensor = m_session->Run(run_option, m_szInputNames.data(), input_onnx.data(), m_szInputNames.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();


        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        auto encoder_out_lens = outputTensor[1].GetTensorMutableData<int64_t>();
        result = greedy_search(floatData, *encoder_out_lens);
    }
    catch (...)
    {
        result = "";
    }
    gettimeofday(&fwd_end, NULL);
    seconds = (fwd_end.tv_sec - fwd_start.tv_sec);
    taking_micros = ((seconds * 1000000) + fwd_end.tv_usec) - (fwd_start.tv_usec);
    total_fwd_time += taking_micros;

    if(in)
        delete in;

    return result;
}

string ModelImp::forward_chunk(float* din, int len, int flag)
{

    printf("Not Imp!!!!!!\n");
    return "Hello";
}

string ModelImp::rescoring()
{
    printf("Not Imp!!!!!!\n");
    return "Hello";
}
