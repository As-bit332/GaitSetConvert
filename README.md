# GaitSetConvert
Convert GaitSet from pytorch to onnx and quantize it for AMD PC AI  
GaitSet is a **flexible**, **effective** and **fast** network for cross-view gait recognition. The source code is in (https://github.com/AbnerHqC/GaitSet)  
The codes above convert the GaitSet model from pytorch to onnx and quantize it for deployment.  

#### Download CASIA-B
http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp  
Download the dataset and use pretreatment.py to crop the silhouette to 64*64  

#### Prepare
Follow the guide in (https://ryzenai.docs.amd.com/en/latest/inst.html) to prepare the environment and conda activate <name>.  

#### Convert Model
Use toonnx.py to generate onnx model. Make sure that transform() in  model/model.py has "pre_process = False, post_process = False".  

#### ModelPreprocess
Use modelpreprocess.py to preprocess the onnx model generated above.  

#### Quantize
Use quantize.py to quantize the onnx model.  

#### Deployment
First change transform() in  model/model.py to have "pre_process = True, post_process = False".  
Then run generate_input.py to generate input data.  
After that, run deploy.py which uses PC AI to generate output data.  
Finally, change transform() in  model/model.py to have "pre_process = False, post_process = True", and run test.py to see the results.
