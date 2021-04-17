# PPQI.jl
Julia Version PaddlePaddle Quick Inference.

# API Reference
* load_config:

  ```julia
  function load_config(
    modelpath::String, 
    use_gpu::Bool=false, 
    gpu_id::Int=0, 
    use_mkldnn::Bool=false, 
    cpu_treads::Int=1
  )::PyObject
    return config::PyObject
  ```

* load_model:

  ```julia
  function load_model(
    config
  )::InferenceModel
    return model::InferenceModel
  ```

* model_forward:

  ```julia
  function model_forward(
    model::InferenceModel, 
    input_datas::Any
  )::Any
    return output_datas::Any
  ```
