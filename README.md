# BDANet: Binary-dimensional Aware Network with multi-wise attention for Cognitive Workload Recognition

## Introduction

Despite significant advancements, EEG data encapsulation of cognitive workload involves both global and local dimensions. Traditional singular network architectures often fall short in fully harnessing the potential of EEG data features. The quest for optimally leveraging these dual-dimensional features in Cognitive Workload Recognition (CWR) remains an ongoing challenge.

## Innovations and Related Works

Recent innovations like EEG_Conformer [21] and LGNet [7] have demonstrated promising results by integrating CNNs with Transformer encoders and employing parallel representation modules for feature fusion. However, these models do not completely actualize the temporal feature extraction capabilities essential for EEG signals.

## BDANet: Addressing the Challenge

To address these challenges, we introduce BDANet, a novel binary-dimensional aware network tailored for Cognitive Workload recognition. This network acknowledges that variations in cognitive workload are influenced by the complexity of current tasks and the cumulative effects of previous tasks. And loacl-global features are considered.

### Core Components of BDANet:

- **BiLSTMs for Temporal Dynamics:** Utilizes two BiLSTMs to effectively capture forward and reverse time series dynamics, offering a robust method for continuous EEG signal analysis.
- **CNN for Spatial Pattern Recognition:** Continues to employ CNN architecture to identify spatial patterns and short-duration features within EEG signals.
- **Multi-wise Attention Mechanism:** Enhances sensitivity to spatio-temporal information through:
  - **Channel-wise Attention:** Weighs the importance of different EEG electrode channels and amplifies inter-channel relationships.
  - **BiLSTM Multi-wise Attention:** Leverages distant dependencies within BiLSTM layers for a more comprehensive representation.
  - **Convolutional Channel Attention:** Recalibrates and models dependencies processed by CNN layers, enhancing the overall efficacy of the network.

## Model Architecture

Below is the architecture diagram of BDANet, illustrating the integration of various components:

![BDANet Model Architecture](网络结构.png)

## Achievements

This innovative approach enables precise identification of complex patterns associated with cognitive loads, thereby enhancing both the accuracy and robustness of the model in real-world applications. BDANet offers a potent tool for the real-time monitoring and evaluation of cognitive workloads. It achieves the state-of-art performance.(91%+)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or newer
- torch 2.1
- This project depends on several key libraries, including PyTorch, NumPy, and Matplotlib. You need to install them by pip or conda.

## Quick Start

To get started with BDANet, follow these steps:

```bash
git clone https://github.com/prestyan/BDANet.git
cd your-repository-name
python main.py // It is BDANet
python main_LG // It is LGNet, according to the author of LGNet, The replicated code may not be publicly available.
```
## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact Information
For any queries or further information, please reach out through the following:

Email: shaoyang@nuaa.edu.cn
GitHub Issues: https://github.com/prestyan/BDANet/issues
