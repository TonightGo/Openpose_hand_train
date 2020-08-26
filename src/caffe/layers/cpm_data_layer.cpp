#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <string>

#include "caffe/common.hpp"
#include "caffe/layers/cpm_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
CPMDataLayer<Dtype>::CPMDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.cpmdata_param().backend()));
  db_->Open(param.cpmdata_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
CPMDataLayer<Dtype>::~CPMDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void CPMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  // Read a data point, and use it to initialize the top blob.
  //Datum& datum = *(reader_.full().peek());
  //LOG(INFO) << datum.height() << " " << datum.width() << " " << datum.channels();

  // Datum datum;
  // datum.ParseFromString(cursor_->value());
  // // Use data_transformer to infer the expected blob shape from datum.
  // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);

  vector<int> top_shape(4);

  // image
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();
  const int height = this->layer_param_.transform_param().crop_size_y();
  const int width = this->layer_param_.transform_param().crop_size_x();
  const bool put_gaussian = this->layer_param_.transform_param().put_gaussian();

  if(put_gaussian){
    top[0]->Reshape(batch_size, 4, height, width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(batch_size, 4, height, width);
    }
    this->transformed_data_.Reshape(1, 4, height, width);
  }
  else {
    top[0]->Reshape(batch_size, 3, height, width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(batch_size, 3, height, width);
    }
    this->transformed_data_.Reshape(1, 3, height, width);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  if (this->output_labels_) {
    const int stride = this->layer_param_.transform_param().stride();
    int num_parts = this->layer_param_.transform_param().num_parts();

    top[1]->Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride); //plus 1 for background
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    }
    this->transformed_label_.Reshape(1, 2*(num_parts+1), height/stride, width/stride);

    LOG(INFO) << "output label size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
  }
}

// template <typename Dtype>
// bool CPMDataLayer<Dtype>::Skip() {
//   int size = Caffe::solver_count();
//   int rank = Caffe::solver_rank();
//   bool keep = (offset_ % size) == rank ||
//               // In test mode, only rank 0 runs, so avoid skipping
//               this->layer_param_.phase() == TEST;
//   return !keep;
// }

template<typename Dtype> void CPMDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void CPMDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double deque_time = 0;
  double read_time = 0;
  // double decod_time = 0;
  double trans_time = 0;
  static int cnt = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  Datum datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    // while (Skip()) {
    //   Next();
    // }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();
    deque_time += timer.MicroSeconds();

    // if (item_id == 0) {
    //   // Reshape according to the first datum of each batch
    //   // on single input batches allows for inputs of varying dimension.
    //   // Use data_transformer to infer the expected blob shape from datum.
    //   vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    //   this->transformed_data_.Reshape(top_shape);
    //   // Reshape batch according to the batch_size.
    //   top_shape[0] = batch_size;
    //   batch->data_.Reshape(top_shape);
    // }

    // Apply data transformations (mirror, scale, crop...)
    // timer.Start();
    const int offset_data = batch->data_.offset(item_id);
    const int offset_label = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    this->transformed_label_.set_cpu_data(top_label + offset_label);

    this->data_transformer_->Transform_CPM(datum, 
        &(this->transformed_data_),
        &(this->transformed_label_), cnt);
    ++cnt;

    trans_time += timer.MicroSeconds();
    Next();
    timer.Stop();
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

// #ifdef BENCHMARK_DATA
//   LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//   LOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms.";
//   LOG(INFO) << "   Decode time: " << decod_time / 1000 << " ms.";
//   LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
// #endif
}

INSTANTIATE_CLASS(CPMDataLayer);
REGISTER_LAYER_CLASS(CPMData);

}  // namespace caffe