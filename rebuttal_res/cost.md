## Cost comparsion of Backdoor and Blackbox attacks

|               | Training Cost (s) | Inference Time (s) | Inference Query |
| ------------- | ----------------- | ------------------ | --------------- |
| CLEAN         | 1124              | 0                  | 0               |
| BELT          | 1242              | 0                  | 0               |
| Ours          | 1768              | 0                  | 0               |
| C&W(Blackbox) | 0                 | 4595               | 10^8            |


while CLEAN and BELT models require 1124 s and 1242 s for training, respectively, our method requires 1768 s. In contrast,  C\&W have no training cost but require significant inference time and queries- C\&W requires 4595 s and $10^8$ queries. 
