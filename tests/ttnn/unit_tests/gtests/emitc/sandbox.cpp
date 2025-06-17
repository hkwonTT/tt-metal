// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/common_values.hpp"
#include "emitc.hpp"

#include "ttnn/operations/point_to_point/point_to_point.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace sandbox {


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
ttnn::Tensor p2PTest(ttnn::Tensor tensor0, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  std::cout << "Getting Global Semaphore for Ccl Ops" << std::endl;
  auto semaphore = ttnn::global_semaphore::create_global_semaphore(
      meshDevice.get(),
      meshDevice.get()->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
      0,                            // initial value
      tt::tt_metal::BufferType::L1  // buffer type
  );
  std::cout << "Got Global Semaphore for Ccl Ops" << std::endl;

  auto coord0 = ttnn::MeshCoordinate(0, 0);
  auto coord1 = ttnn::MeshCoordinate(0, 1);
  std::cout << "Running Point to Point" << std::endl;
  ttnn::Tensor organizedTensor = ttnn::point_to_point(tensor0, coord0, coord1, ::ttnn::ccl::Topology::Linear, semaphore);
  std::cout << "Finished Point to Point" << std::endl;
  return organizedTensor;
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
  size_t stride = 1;
  int targetDim = 1;
  size_t iteration = 2;
  std::vector<::ttnn::Tensor> target_tensors;
  for (size_t i = 0; i < iteration; ++i) {
    target_tensors.push_back(input_tensors[i * stride]);
  }
  return ::ttnn::experimental::xtensor::concat(target_tensors, targetDim);
}
void sandbox(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput1 = shardTensor(inputTensor, meshDevice);
  // ttnn::Tensor shardedInput2 = shardTensor(inputTensor, meshDevice);
  // ttnn::Tensor shardedOutput = sandboxFunction(shardedInput1, shardedInput2, meshDevice);
  ttnn::Tensor shardedOutput = p2PTest(shardedInput1, meshDevice);
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}

TEST(EmitC, Sandbox) {
  tt::tt_metal::MetalContext::instance().initialize(tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::ETH}, 1, {});
  tt::tt_metal::MetalContext::instance().set_fabric_config(tt::tt_metal::FabricConfig::FABRIC_1D);
  
  std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice = sandbox::openMeshDevice();

  sandbox::sandbox(meshDevice);
}
}