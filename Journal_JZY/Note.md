常规conda environment的建立流程：

    conda create -n llm-from-scratch python=3.10

    conda actiavte llm-from-scratch

    pip install -r requirements.txt

    conda install jupyter ipykernel

    python -m ipykernel install --user --name=llm-from-scratch --display-name="llm-from-scratch"


然后就是check pytorch的安装情况：

import torch 
# Check PyTorch version 
print(f'PyTorch version: {torch.__version__}')

# 也可以用pip来在terminal里直接check 
# pip list | grep torch

# Check if CUDA is available and which version
print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "Not available"}')

# Check Related Pytorch Packages
# Check for torchvision, torchaudio, etc.
import torch; import torchvision; import torchaudio; 
print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}, torchaudio: {torchaudio.__version__}')


# Verify your installation works perfectly
# Create a simple test script
import torch; x = torch.rand(5, 3) 
print(x)



常见问题之debug CUDA:

万一找不到cuda：(解释：一般直接pip install torch torchvision torchaudio似乎就容易装到只是cpu的版本，好像cuda的版本得跑那种pip install里有一个网站的才行, 而且得确定装的是有-cuda几个字的版本)

Step 1:
  在terminal里跑：nvidia-smi

  找到支持的CUDA最高版本（我的电脑应该是CUDA Version: 12.7），然后上官网去找对应的下载的代码

Step 2:
  把之前的torch版本卸载掉：
      pip uninstall torch torchvision torchaudio 
      选yes (输入y)

  # completely remove any existing (probably CPU-only) builds of PyTorch before installing a fresh, CUDA-enabled wheel 
  pip cache purge 

Step 3: 安装带cuda的pytorch
# 用conda安装的话就用：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 用pip安装的话就用：（貌似pip安装不太好用）

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


可以尝试用Mambda加速：
conda install mamba -n base -c conda-forge
然后之后可以用mambda install ....

Step 4:
如果系统还是检测不到CUDA，可能是CUDA toolkit没装，可以尝试 Install CUDA toolkit via conda：
conda install cudatoolkit=11.8 -c nvidia

Step 5: Verify the installation

在terminal里：
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"


或者直接在ipynb的code chunk里：
  import torch 
  # Check PyTorch version 
  print(f'PyTorch version: {torch.__version__}')

  # Check if CUDA is available and which version
  print(f'CUDA available: {torch.cuda.is_available()}')
  print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "Not available"}')



2025-05-17
今天学了什么：nn.Module 
super().__init__()
  得有这一行才可以explicitly call parent class's initialization method,不然parent class的initialization里所有定义的东西都不会出现和不能用


2025-05-18 

今天得把nn.Module再学明白一点，把nn.Module里面常见的那几个功能的源代码看了，比如说.to(device), forward(), .parameters(), .state_dict(), .load_state_dict(), .eval(), .train(), .zero_grad(), .backward(), .step(), __call__(), 了解nn.Module是怎么track parameters, buffers, modules and etc.  

今天学到的东西：
1. nn.ModuleList()  : ModuleList can be indexed like a regular python list, but modules it contains are properly registered, and will be visible by all Module methods. 
2. List Comprehension: 
  Basic syntax: new_list = [expression for item in iterable if condition]

3. 我现在build的那些instance（比如说CausalAttention, FeedForward, Embedding, etc.），都是继承了nn.Module的，所以它们都是可以当作一个module来用的（换句话说它们都是module, 比如说我们会说CausalAttention is a module, FeedForward is a module, Embedding is a module, 然后具体的intsance我们叫做module instance, 比如说'ca' is a module instance contains trainable parameters and buffers），所以可以：
   model = Model()
   model.to(device)
   model.eval()
   model.train()
   model.zero_grad()
   model.backward()
   model.step()

  4. What "Properly registered" means:
    "Registration" connects a module to PyTorch's parameter management system:
  
  ```python
  class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # These are registered automatically:
        self.linear = nn.Linear(10, 10)  # ✓ Registered
        self.module_list = nn.ModuleList([nn.Linear(5, 5)])  # ✓ Registered
        
        # These are NOT registered:
        self.normal_list = [nn.Linear(10, 10)]  # ✗ Not registered
        self.some_layer = None  # ✗ Not registered
        linear2 = nn.Linear(10, 10)  # ✗ Not registered (local variable)
  ```
  When a module is "registered," it means:

  - Its parameters appear in model.parameters() (for optimizers)
  - It's included in model.state_dict() (for saving)
  - It responds to model.to(device) (for GPU usage)
  - It switches modes with model.train() and model.eval()

5. Module Methods
"Module methods" refer to functions inherited from nn.Module:

parameters(): Gets all trainable parameters
state_dict(): Exports model state
load_state_dict(): Imports model state
to(device): Moves to CPU/GPU
train(): Sets training mode
eval(): Sets evaluation mode

6. Module Definition vs. Execution 






到底什么是nn.module and how the registration system of nn.module works? （可以看gemini的notes! gemini的： Pytorch nn.Module Registration system）
大概可以理解成：比如run nn.Linear()这个class的时候，你的initalization里其实包含了self.weights = nn.Parameter(torch.rand(in_features, out_features)) 还有self.bias = nn.Parameter(torch.rand(out_features)), 你super().__init__()的时候，你会顺便call self.Parameter() (对应torch.nn.Parameter)，

然后你就可以在self.parameters()里看到self.weights 和 self.bias了，然后你就可以用optimizer.step()来更新self.weights 和 self.bias了。



今天学了OOP的一些概念， 比如说instance.attribute = value 的 python底层的execution到底是怎么样的，当出现这一行的时候，python其实是在run type(instance).__setattr__(instance, "attribute", value)， 于是牵扯到了为什么我们会在Pytorch里重新overwrite__setattr__() ， 这样才可以满足pytorch的一些功能。

Whenever you execute a line of the form:
my_object.some_attribute = some_value

Python internally performs an operation that is roughly equivalent to:
type(my_object).__setattr__(my_object, "some_attribute", some_value)




可以补一下python的"data model" 什么的， 比如说__init__, __setattr__, __getattr__, __call__, __len__, etc. 
This is a fundamental part of Python's "data model" or "object model." Special methods (often called "dunder" methods, for double underscores, like __init__, __setattr__, __getattr__, __call__, __len__, etc.) allow classes to customize the behavior of built-in Python operations like attribute access, function calling, length checking, etc.


注：我怕自己nn.Module学太深，今天先学到这里。回头有空再好好总结Gemini今天的Notes，今天的notes很重要，只要把registration system搞明白，nn.Module的很多功能也就可以理解了。但今天任务是把chap 3 给手搓完，还有chap 4也尽量手搓完！先去看chap 3 和 chap 4 chap 5的code. 





