from onnxruntime.quantization import shape_inference

shape_inference.quant_pre_process(
   input_model_path = "work/ryzenai_gaitset.onnx",
   output_model_path = "work/ryzenai_gaitset_preprocessed.onnx",
   skip_optimization= False,
   skip_onnx_shape = False,
   skip_symbolic_shape= False,
   auto_merge = False,
   int_max=2**31 - 1,
   guess_output_rank = False,
   verbose = 0,
   save_as_external_data = False,
   all_tensors_to_one_file = False,
   external_data_location = "./",
   external_data_size_threshold = 1024,)