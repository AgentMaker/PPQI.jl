__precompile__(true)

module PPQI

using PyCall

export load_config, load_model, model_forward

const inference = PyNULL()

function __init__()
    copy!(inference, pyimport("paddle.inference"))
end


struct InferenceModel
    predictor::PyObject
    input_handles::Vector{PyObject}
    output_handles::Vector{PyObject}
end


function load_config(modelpath::String, use_gpu::Bool=false, gpu_id::Int=0, use_mkldnn::Bool=false, cpu_treads::Int=1)::PyObject
    if isdir(modelpath)

        if isfile(joinpath(modelpath, "__params__"))
            model = joinpath(modelpath, "__model__")
            params = joinpath(modelpath, "__params__")
            config = inference.Config(model, params)

        elseif isfile(joinpath(modelpath, "params"))
            model = joinpath(modelpath, "model")
            params = joinpath(modelpath, "params")
            config = inference.Config(model, params)

        elseif isfile(joinpath(modelpath, "__model__"))
            config = inference.Config(modelpath)

        end

    elseif isfile(modelpath + ".pdmodel")
        model = modelpath + ".pdmodel"
        params = modelpath + ".pdiparams"
        config = inference.Config(model, params)

    end

    if use_gpu
        config.enable_use_gpu(100, gpu_id)

    else
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_treads)

        if use_mkldnn
            config.enable_mkldnn()
        end

    end

    return config

end


function load_model(config)::InferenceModel
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_handles = PyObject[]

    for input_name in input_names
        input_handle = predictor.get_input_handle(input_name)
        push!(input_handles, input_handle)
    end

    output_handles = PyObject[]

    for output_name in output_names
        output_handle = predictor.get_output_handle(output_name)
        push!(output_handles, output_handle)
    end

    model = InferenceModel(predictor, input_handles, output_handles)

    return model
end


function model_forward(model::InferenceModel, input_datas::Vector{Array})::Vector{PyObject}
    for input_handle in model.input_handles, data in input_datas
        input_handle.copy_from_cpu(data) 
    end

    model.predictor.run()
    output_datas = PyObject[]

    for output_handle in model.output_handles
        data = output_handle.copy_to_cpu() 
        push!(output_datas, data)
    end

    return output_datas 

end

end
