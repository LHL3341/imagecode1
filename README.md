# Imagecode

1. clone

2. 解压所有zip文件

3. clone OFA代码和OFA-large

   ```bash
   git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
   pip install OFA/transformers/
   git clone https://huggingface.co/OFA-Sys/OFA-large
   ```


4. 下载OFA-large(https://huggingface.co/OFA-Sys/ofa-large)中的pytorch_model.bin，
   放于OFA-large/目录下
6. 下载数据集，放在data/目录下
7. 配置环境
