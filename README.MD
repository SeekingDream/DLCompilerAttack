# `📖 Dcl-BD`
This code repository contains the main implementation of **Dcl-BD**, introduced in our paper, *"Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers."*
Dcl-BD demonstrates how fundamental flaws in the design of popular deep learning (DL) compilers can be leveraged to create models that appear completely benign before compilation but become maliciously backdoored after passing through a compiler.

This repository provides:

* Core code for constructing and evaluating models using the Dcl-BD attack pipeline
* Scripts for reproducing our experiments and benchmarks from the paper
* Example models and instructions for testing across various DL compilers

Our work reveals a new attack surface in the machine learning systems, a new system level attack, highlighting the urgent need to address compilation-induced vulnerabilities in real-world AI systems.

## Attack Scenario Overview
<div  align="center">    
 <img src="https://github.com/SeekingDream/DLCompilerAttack/blob/main/figs/Dcl-BD.jpg" width="700" height="350" alt="Design Overview"/><br/>
</div>   

The above figure shows our attack scenario,  the attacker will perform the following steps to exploit the semantic inconsistent vulnerability in deep learning compielr to launch the stealthy attack:

**Step ① Publish:**
The attacker acts as a DL model provider and uploads a seemingly benign pre-trained model to public model platforms, such as HuggingFace, GitHub, or Model Zoo.

**Step ② Download:**
Victims (developers or organizations) download these models for use in their own applications.

**Step ③ Model Checking:**
Before deployment, victims typically verify the downloaded model’s security and accuracy to ensure it behaves as expected.

**Step ④ Compilation:**
To enable efficient, real-time inference—especially on mobile or resource-constrained devices—the victim compiles the model using a commercial DL compiler (e.g., TVM, ONNX Runtime, TensorRT). This conversion is often necessary for compatibility with target hardware.

**Step ⑤ Backdoor Attack:**
Our work demonstrates that an attacker can exploit the compilation process. The model appears benign before compilation, passing all security checks. However, after compilation, a hidden backdoor is activated, allowing the attacker to trigger malicious behavior in the deployed model.

## File Structure
+ environment.yml           # Environment configuration file for setting up dependencies.
+ huggingface_model.txt     # The model name list in our in-the-wild evaluation.
+ utils.py                  # Basic utility functions.
+ main.py                   # Main script to launch the attack and train models.
+ wild_evaluation.py        # Scripts for evaluating attacks on HuggingFace models.
+ transferability.py        # Evaluates the transferability of the attack across models.
+ train_model_clean.py            # Script for training a CLEAN (benign) model.
+ run_detector.py           # Main script to run various backdoor detectors.
+ prediction.py             # Inference script for testing models on evaluation datasets.
+ finetune_defense.py       # Fine-tunes attacked models as a defense strategy.

+ src/methods/              # Core implementation directory.
  + attack/                 # Implements attack algorithms.
  + detector/               # Implements detection algorithms.
  + model/                  # Implements model architectures.
  + dlcl.py                 # Abstract class for deep learning compiler interfaces.
  + abst_cl_model.py        # Abstract class for compiled models, providing a uniform inference API.


## How to Run


### 1. Set Up the Environment

Run the following command to create the environment specified in `environment.yml`:

```bash
conda env create -f environment.yml
```

---

### 2. Train a CLEAN Model

To train a clean model for task 0, use:

```bash
python train_model_clean.py --task_id 0
```

---

### 3. Launch the Attack

To launch the attack, run:

```bash
python main.py --task_id 0 --cl_id 0 --hardware_id 0
```

### Reproduce Our Main Results

```
bash reproduce_main.bash
```

* `task_id` specifies the model:
  * `0`: C10-CN               (CIFAR10 ConvNet)
  * `1`: C10-V16              (CIFAR10 VGG16) 
  * `2`: C100-R18             (CIFAR100 ResNet18)
  * `3`: C100-V19             (CIFAR100 VGG19)
  * `4`: Tiny-R34             (TinyImgNet ResNet34)
  * `5`: Tiny-RX29            (TinyImgNet ResXNet29)

* `cl_id` specifies the compiler to use:
  * `0`: torch.compile
  * `1`: TVM
  * `2`: ONNXRuntime

* `hardware_id` specifies the deployment hardware:

  * `0`: GPU
  * `-1`: CPU
    
## Supported Deep Learning Compilers and Hardware Backends

| Compiler      | CPU Support | GPU Support |
|---------------|:-----------:|:-----------:|
| [torch.compile](https://github.com/pytorch/pytorch) |     ✅      |     ✅      |
| [onnxruntime](https://github.com/microsoft/onnxruntime)   |     ✅      |     ✅      |
| [TVM](https://github.com/apache/tvm)           |     ✅      |     ✅      |
| [tensorRT](https://github.com/NVIDIA/TensorRT)      |             |     ✅      |
| [MLIR](https://github.com/iree-org/iree)          |     ✅      |           |


## In-the-wild Study Model List

The table below lists our selected in-the-wild models, chosen from the top 100 most popular models on HuggingFace. The complete list of model names can be found in [huggingface\_model.txt](https://github.com/SeekingDream/DcL-BD/blob/main/huggingface_model.txt).


| #  | Model Name                                 |
|----|--------------------------------------------|
| 1  | microsoft/resnet-50                        |
| 2  | timm/mobilenetv3_small_100.lamb_in1k       |
| 3  | timm/resnet50.a1_in1k                      |
| 4  | Falconsai/nsfw_image_detection             |
| 5  | google/vit-base-patch16-224                |
| 6  | WinKawaks/vit-tiny-patch16-224             |
| 7  | nateraw/vit-age-classifier                 |
| 8  | rizvandwiki/gender-classification          |
| 9  | timm/resnet18.a1_in1k                      |
| 10 | microsoft/beit-base-patch16-224-pt22k-ft22k|


## Generalized Attack API

The original entry point (`main.py`) dispatches by `task_id` (0–5): each id maps
to a fixed (model, dataset, wrapper) combination, and the model-splitting logic
is hardcoded in `load_attack_model_cls`. The `Attacker` API removes that
dispatch — you hand it a model and a dataset, and it selects how to split the
model automatically:

```python
from src.api import Attacker

model = load_model(0, load_pretrained=False)      # any supported model
attacker = Attacker(model, dataset=(train_loader, test_loader))
attacker.run()                                     # 3-stage attack (needs GPU + compiler)
```

Runnable example: `examples/general_attack_example.py`.

### How it works

- **Adapters** (`src/adapters.py`) know how to split *one family* of models into
  the front half (`FeatureModel`) and back half (`TunedModel`) the attack
  consumes. Each adapter is thin — it just reuses the existing wrapper classes.
- **`Attacker`** (`src/api.py`) looks up the adapter for a model by type
  (`REGISTRY`), splits the model, auto-computes the embedding shape at the split
  (by running one dummy forward pass — no hardcoded shapes), and hands the halves
  to the **unmodified** `DLCompilerAttack` engine.

### What is supported

- The four CNN families the paper evaluates: **ConvNet, VGG, ResNet, ResNeXt**
  (i.e. `task_id` 0–5), now reachable through the `Attacker` API without a
  `task_id`.
- The old path is untouched: `python main.py --task_id 0..5` still works exactly
  as before.

### What is NOT supported (by design)

- **Automatic splitting of arbitrary models.** Where to cut a model is a semantic
  choice the code cannot reliably infer. For an unrecognized model, write a small
  adapter and pass it in:

  ```python
  from src.adapters import Adapter
  class MyAdapter(Adapter):
      def make_front_half(self, model): ...
      def make_back_half(self, model, embed_shape): ...
  attacker = Attacker(my_model, dataset, adapter=MyAdapter())
  ```

- **Non-CNN architectures** (e.g. ViT/Swin). `utils.load_model` can *load* them
  (task 6/7), but they have no attack wrappers and are outside the paper's scope.
- **Running a full attack on a laptop.** `run()` needs a GPU and a real compiler
  backend. Without them you can still assemble the engine (`build_attacker()`)
  and run the smoke tests in `tests/`.

### Running on Apple Silicon / machines without TVM

`import src` pulls in modules that `import tvm`, which has no Apple Silicon build.
`mac_tvm_stub.py` installs a stand-in for `tvm` *only when the real one is
missing* (a no-op on Simin's CUDA box). Import it before `src`:

```python
import mac_tvm_stub   # must come before `import src`
```

### Smoke tests

```bash
python tests/smoke_families.py   # all 4 CNN families (task 0–5)
python tests/smoke_assemble.py   # Attacker assembles the DLCompilerAttack engine
```


