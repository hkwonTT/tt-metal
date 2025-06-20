// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include "hostdevcommon/common_values.hpp"
#include "emitc.hpp"

#include "ttnn/operations/point_to_point/point_to_point.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace sandbox {


void printTensor(ttnn::Tensor &tensor, std::string name = "UNKNOWN") {
  std::cout << "Tensor: " << name << std::endl;
  std::cout << "  Shape: " << tensor.logical_shape()[0] << ", " << tensor.logical_shape()[1] << std::endl;
  auto storageType2Str = [](ttnn::StorageType storageType) {
    switch (storageType) {
      case ttnn::StorageType::DEVICE:
        return "DEVICE";
      case ttnn::StorageType::HOST:
        return "HOST";
      case ttnn::StorageType::MULTI_DEVICE_HOST:
        return "MULTI_DEVICE_HOST";
    }
  };
  std::cout << "  Storage Type: " << storageType2Str(tensor.storage_type()) << std::endl;
  if (tensor.storage_type() == ttnn::StorageType::MULTI_DEVICE_HOST) {
    auto bufferShape = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage()).distributed_buffer().shape();
    std::cout << "  Buffer Shape: " << bufferShape[0] << ", " << bufferShape[1] << std::endl;
  } else if (tensor.storage_type() == ttnn::StorageType::DEVICE) {
    auto bufferShape = std::get<tt::tt_metal::DeviceStorage>(tensor.get_storage()).get_mesh_buffer()->device()->shape();
    std::cout << "  Buffer Shape: " << bufferShape[0] << ", " << bufferShape[1] << std::endl;

  }
}

ttnn::Tensor aggretateTensors(std::vector<ttnn::Tensor> tensors, const tt::tt_metal::DistributedTensorConfig& config) {
  std::cout << "Aggregating Tensors" << std::endl;
  assert(tensors.size() > 0);
  ttnn::Tensor& refTensor = tensors[0];
  for (auto& tensor : tensors) {
    assert(tensor.storage_type() == refTensor.storage_type());
  }
  bool isDeviceStorage = refTensor.storage_type() == ttnn::StorageType::DEVICE;

  if (isDeviceStorage) {
    bool isSameMeshBuffer = true;
    auto refMeshBuffer = std::get<tt::tt_metal::DeviceStorage>(refTensor.get_storage()).mesh_buffer;
    for (auto& tensor : tensors) {
      if (refMeshBuffer != std::get<tt::tt_metal::DeviceStorage>(tensor.get_storage()).mesh_buffer) {
        isSameMeshBuffer = false;
        break;
      }
    }
    if (!isSameMeshBuffer) {
      std::vector<ttnn::Tensor> hostTensors;
      for (auto& tensor : tensors) {
        auto hostTensor = ttnn::from_device(tensor);
        hostTensors.push_back(ttnn::Tensor(tt::tt_metal::host_buffer::get_host_buffer(hostTensor), hostTensor.tensor_spec()));
      }
      auto aggregatedTensor = ttnn::distributed::aggregate_as_tensor(hostTensors, config);
      return ttnn::to_device(aggregatedTensor, refTensor.mesh_device(), std::nullopt);
    }
  }
  return ttnn::distributed::aggregate_as_tensor(tensors, config);
}

size_t getCoordsCount(ttnn::Tensor tensor) {
    return std::get<tt::tt_metal::DeviceStorage>(tensor.storage()).coords.size();
}

ttnn::Tensor showCoordsOnDeviceTensor(ttnn::Tensor shardTensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice, ttnn::GlobalSemaphore semaphore) {

    ttnn::Tensor extractedShardTensor = ttnn::distributed::get_device_tensors(shardTensor)[0];
    ttnn::Tensor p2pOutputTensor = ttnn::point_to_point(shardTensor, ttnn::MeshCoordinate(0, 0), ttnn::MeshCoordinate(0, 1), ::ttnn::ccl::Topology::Linear, semaphore);

    std::cout << "Tensor sharded : " << getCoordsCount(shardTensor) << std::endl;
    std::cout << "Tensor from get_device_tensors : " << getCoordsCount(extractedShardTensor) << std::endl;
    std::cout << "Tensor from p2p : " << getCoordsCount(p2pOutputTensor) << std::endl;

    return shardTensor;
}

ttnn::Tensor pointToPoint(ttnn::Tensor srcTensor, ttnn::MeshCoordinate srcCoord, ttnn::MeshCoordinate dstCoord, ::ttnn::ccl::Topology topology, ttnn::GlobalSemaphore semaphore) {
#if 1
  std::vector<ttnn::Tensor> hostTensors = ttnn::distributed::get_device_tensors(srcTensor);
  std::vector<ttnn::Tensor> p2pOutputTensors;
  for (size_t idx = 0; idx < hostTensors.size(); idx++) {
    
  }
    
    p2pOutputTensors.push_back(p2pOutputTensor);
  }
  return ttnn::distributed::aggregate_as_tensor(p2pOutputTensors, srcTensor.distributed_tensor_config());

  
#else
  return ttnn::point_to_point(srcTensor, srcCoord, dstCoord, topology, semaphore);  
#endif
}

ttnn::Tensor p2PTest(ttnn::Tensor tensor0, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  std::cout << "Getting Global Semaphore for Ccl Ops" << std::endl;
  auto semaphore = ttnn::global_semaphore::create_global_semaphore(
      meshDevice.get(),
      meshDevice.get()->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
      0,                            // initial value
      tt::tt_metal::BufferType::L1  // buffer type
  );
  std::cout << "Got Global Semaphore for Ccl Ops" << std::endl;
  auto full = tensor0.logical_shape()[1];
  auto half = full / 2;
  std::cout << "input tensor shape : " << tensor0.logical_shape()[0] << ", " << tensor0.logical_shape()[1] << std::endl;
  
  ::ttnn::SmallVector<int32_t> begins({0, 0});
  ::ttnn::SmallVector<int32_t> ends({256, half});
  ::ttnn::SmallVector<int32_t> step({1, 1});
  std::cout << "Slicing 0" << std::endl;
  ::ttnn::Tensor slice_0 = ::ttnn::slice(tensor0, begins, ends, step);

  begins[1] = half;
  ends[1] = full;
  std::cout << "Slicing 1" << std::endl;
  ::ttnn::Tensor slice_1 = ::ttnn::slice(tensor0, begins, ends, step);

  auto coord0 = ttnn::MeshCoordinate(0, 0);
  auto coord1 = ttnn::MeshCoordinate(0, 1);

  std::cout << "get_device_tensors for slice_0" << std::endl;
  auto tensorsSlice0 = ttnn::distributed::get_device_tensors(slice_0);
  auto tensorsSlice1 = ttnn::distributed::get_device_tensors(slice_1);
  for (auto& tensor : tensorsSlice0) {
    printTensor(tensor, "Slice 0");
  }
  for (auto& tensor : tensorsSlice1) {
    printTensor(tensor, "Slice 1");
  }
  ttnn::Tensor split_0_0 = tensorsSlice0[0];
  std::cout << "Running Point to Point 1 to 0" << std::endl;
  ttnn::Tensor split_1_0 = ttnn::point_to_point(slice_0, coord1, coord0, ::ttnn::ccl::Topology::Linear, semaphore);

  auto storage_shard = std::get<tt::tt_metal::DeviceStorage>(tensor0.storage());
  auto storage_get_device_tensor = std::get<tt::tt_metal::DeviceStorage>(split_0_0.storage());
  auto storage_p2p_tensor = std::get<tt::tt_metal::DeviceStorage>(split_1_0.storage());
  std::cout << "Tensor from sharding : " << storage_shard.coords.size() << std::endl;
  std::cout << "Tensor from get_device_tensors : " << storage_get_device_tensor.coords.size() << std::endl;
  std::cout << "Tensor from p2p : " << storage_p2p_tensor.coords.size() << std::endl;



  
  std::cout << "Running Point to Point 0 to 1" << std::endl;
  ttnn::Tensor split_0_1 = ttnn::point_to_point(slice_1, coord0, coord1, ::ttnn::ccl::Topology::Linear, semaphore);
  std::cout << "get_device_tensors for slice_1" << std::endl;
  ttnn::Tensor split_1_1 = tensorsSlice1[1];
  std::cout << "Finished Point to Point" << std::endl;

  // aggregating on host
  ttnn::Tensor split_0_0_h = ttnn::from_device(split_0_0);
  ttnn::Tensor split_1_0_h = ttnn::from_device(split_1_0);
  ttnn::Tensor split_0_1_h = ttnn::from_device(split_0_1);
  ttnn::Tensor split_1_1_h = ttnn::from_device(split_1_1);
  std::cout << "from_device done" << std::endl;
  printTensor(split_0_0_h);
  printTensor(split_1_0_h);
  printTensor(split_0_1_h);
  printTensor(split_1_1_h);

  std::cout << "tranforming to host tensor for 0,0" << std::endl;
  split_0_0_h = ::ttnn::Tensor(tt::tt_metal::host_buffer::get_host_buffer(split_0_0_h),
                       split_0_0_h.tensor_spec());
  std::cout << "tranforming to host tensor for 1,1" << std::endl;
  split_1_1_h = ::ttnn::Tensor(tt::tt_metal::host_buffer::get_host_buffer(split_1_1_h),
                       split_1_1_h.tensor_spec());
  std::cout << "tranforming to host tensor for 1,0" << std::endl;
  split_1_0_h = ::ttnn::Tensor(tt::tt_metal::host_buffer::get_host_buffer(split_1_0_h),
                       split_1_0_h.tensor_spec());
  std::cout << "tranforming to host tensor for 0,1" << std::endl;
  split_0_1_h = ::ttnn::Tensor(tt::tt_metal::host_buffer::get_host_buffer(split_0_1_h),
                       split_0_1_h.tensor_spec());

  ttnn::Tensor split_0_h = ttnn::distributed::aggregate_as_tensor({split_0_0_h, split_0_1_h}, ::tt::tt_metal::AllGatherTensor());
  std::cout << "aggregate 0 done" << std::endl;
  ttnn::Tensor split_1_h = ttnn::distributed::aggregate_as_tensor({split_1_0_h, split_1_1_h}, ::tt::tt_metal::AllGatherTensor());
  std::cout << "aggregate 1 done" << std::endl;
  ttnn::Tensor split_0 = ttnn::to_device(split_0_h, meshDevice.get(), std::nullopt);
  ttnn::Tensor split_1 = ttnn::to_device(split_1_h, meshDevice.get(), std::nullopt);
  std::cout << "to_device done" << std::endl;

  std::vector splits = {split_0, split_1};

  ttnn::Tensor output = ttnn::concat(splits, 1, std::nullopt);
  std::cout << "concat done" << std::endl;

  
  return output;
}

ttnn::Tensor sandboxFunction(ttnn::Tensor tensor0, ttnn::Tensor tensor1, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  std::cout << "Running Sandbox Function" << std::endl;

  std::vector<std::shared_ptr<ttnn::distributed::MeshDevice>> targetSubmeshes; // store submesh to avoid deconstruction

  std::vector<ttnn::Tensor> deviceTensors0 = ttnn::distributed::get_device_tensors(tensor0);
  std::vector<ttnn::Tensor> deviceTensors1 = ttnn::distributed::get_device_tensors(tensor1);
  std::vector<std::vector<ttnn::Tensor>> reorgTensors(deviceTensors0.size(), std::vector<ttnn::Tensor>(deviceTensors1.size()));
  for (size_t i = 0; i < deviceTensors0.size(); i++)
  {
    std::cout << "Reorganizing tensor from Device #" << i << std::endl;
    auto hostTensor = ttnn::from_device(deviceTensors0[i]);
    std::cout << "Creating unit submesh of Device #" << deviceTensors0.size() - i - 1 << std::endl;
    auto targetDevice = meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, deviceTensors0.size() - i - 1));
    targetSubmeshes.push_back(targetDevice); // store submesh to avoid deconstruction
    reorgTensors[0][i] = ttnn::to_device(hostTensor, targetDevice.get(), std::nullopt);
  }
  for (size_t i = 0; i < deviceTensors1.size(); i++)
  {
    std::cout << "Reorganizing tensor from Device #" << i << std::endl;
    auto hostTensor = ttnn::from_device(deviceTensors1[i]);
    std::cout << "Creating unit submesh of Device #" << deviceTensors1.size() - i - 1 << std::endl;
    auto &targetDevice = targetSubmeshes[i];
    reorgTensors[1][i] = ttnn::to_device(hostTensor, targetDevice.get(), std::nullopt);
  }

  std::vector<ttnn::Tensor> shardedTensors;
  std::cout << "Aggregating single device tensors as a sharded multi device Tensor - 0" << std::endl;
  shardedTensors.push_back(aggretateTensors(reorgTensors[0], tensor0.distributed_tensor_config()));
  std::cout << "Aggregating single device tensors as a sharded multi device Tensor - 1" << std::endl;
  shardedTensors.push_back(aggretateTensors(reorgTensors[1], tensor1.distributed_tensor_config()));
  std::cout << "concat" << std::endl;
  ttnn::Tensor output = ttnn::concat(shardedTensors, 0, std::nullopt);
  std::cout << "Done" << std::endl;
  std::cout << targetSubmeshes.size() << std::endl;
  return output;
}


ttnn::Tensor create_inputs_for_testing() {
  std::cout << "Creating Inputs for Testing" << std::endl;
  ttnn::Tensor v1 =
      ttnn::ones(ttnn::Shape({256, 256}), ttnn::DataType::FLOAT32,
                 ttnn::Layout::ROW_MAJOR, ::std::nullopt,
                 ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ttnn::BufferType::SYSTEM_MEMORY});
  return v1;
}
std::shared_ptr<ttnn::distributed::MeshDevice> openMeshDevice() {
  return ttnn::distributed::open_mesh_device(
      ttnn::MeshShape(1, 2), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE,
      1, tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::ETH},
      std::nullopt, std::vector<int>{}, DEFAULT_WORKER_L1_SIZE);
}

using ttnn::distributed::MeshMapperConfig;
using ttnn::distributed::MeshToTensor;
using ttnn::distributed::TensorToMesh;
ttnn::Tensor shardTensor(ttnn::Tensor inputTensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  std::cout << "Sharding Tensor" << std::endl;
  MeshMapperConfig config{.placements = {MeshMapperConfig::Replicate(),
                                         MeshMapperConfig::Shard(1)}};

  std::unique_ptr<TensorToMesh> meshMapper =
      ::ttnn::distributed::create_mesh_mapper(*meshDevice, config);

  ttnn::Tensor shardedInputHost = ::ttnn::distributed::distribute_tensor(inputTensor, *meshMapper,
                                                *meshDevice);
  std::cout << "Distributing done." << std::endl;

  // ttnn::Tensor shardedInputLayout = ttnn::to_layout(
  //     shardedInputHost, ttnn::Layout::TILE, ::std::nullopt,
  //     ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
  //                          ttnn::BufferType::SYSTEM_MEMORY},
  //     static_cast<ttnn::distributed::MeshDevice *>(nullptr));

  // std::cout << "to_layout done." << std::endl;

  ttnn::Tensor shardedInput = ttnn::to_device(shardedInputHost, meshDevice.get(),
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM});
  std::cout << "to_device done." << std::endl;
  return shardedInput;
}

ttnn::Tensor unshardTensor(ttnn::Tensor shardedTensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  std::cout << "Unsharding Tensor" << std::endl;
  ttnn::Tensor shardedHost = ttnn::from_device(shardedTensor);
  // ttnn::Tensor shardedOutputLayout = ttnn::to_layout(
  //     shardedHost, ttnn::Layout::ROW_MAJOR, ::std::nullopt,
  //     ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
  //                          ttnn::BufferType::SYSTEM_MEMORY},
  //     static_cast<ttnn::distributed::MeshDevice *>(nullptr));
  
  std::vector<::ttnn::Tensor> input_tensors =
      ::ttnn::distributed::get_device_tensors(shardedHost);
  int targetDim = 1;
  return ::ttnn::experimental::xtensor::concat(input_tensors, targetDim);
}
void sandbox(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
    ttnn::GlobalSemaphore semaphore = ttnn::global_semaphore::create_global_semaphore(
        meshDevice.get(),
        meshDevice.get()->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
        0,
        tt::tt_metal::BufferType::L1);
    ttnn::Tensor inputTensor = create_inputs_for_testing();
    ttnn::Tensor shardedInput1 = shardTensor(inputTensor, meshDevice);
    // ttnn::Tensor shardedInput2 = shardTensor(inputTensor, meshDevice);
    // ttnn::Tensor shardedOutput = sandboxFunction(shardedInput1, shardedInput2, meshDevice);
    // ttnn::Tensor shardedOutput = p2PTest(shardedInput1, meshDevice);
    ttnn::Tensor shardedOutput = showCoordsOnDeviceTensor(shardedInput1, meshDevice, semaphore);
    ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}

TEST(EmitC, Sandbox) {
  tt::tt_metal::MetalContext::instance().initialize(tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::ETH}, 1, {});
  tt::tt_metal::MetalContext::instance().set_fabric_config(tt::tt_metal::FabricConfig::FABRIC_1D);
  
  std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice = sandbox::openMeshDevice();

  sandbox::sandbox(meshDevice);
}
}