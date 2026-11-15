# Reconhecimento de Voz (KWS): PTQ vs. QAT (incluindo Sub-8-Bit)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedrohugo-psc/Reconhecimento_Voz_PTQ_vs_QAT/blob/main/Reconhecimento_Voz_PTQ_vs_QAT.ipynb)

- **Instituição:** Universidade Federal da Bahia (UFBA) — Escola Politécnica / Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC)
- **Disciplina:** Otimização (`PPGEE0016`) — `2026.2`
- **Professor:** Prof. Dr. Edmar Egidio Purcino de Souza
- **Aluno:** Pedro Hugo Passos da Silva Carlos
- **Avaliação:** Avaliação II — Trabalho Computacional

---

## Visão Geral

Este repositório reproduz em **PyTorch** o *pipeline* de reconhecimento de comandos de voz (*Keyword Spotting* — KWS) desenvolvido originalmente na plataforma **Edge Impulse** para o microcontrolador ESP32-S3 (`Carlos, 2025`), mantendo exatamente os mesmos parâmetros de extração **MFCC** (`13` coeficientes, quadros/passo de `20 ms`, `32` filtros Mel, FFT `512`, pré-ênfase `0,98`, CMVN `101`) e a arquitetura **1D-CNN** de `1.974` parâmetros treinada em `300` ciclos (`Adam`, `lr = 0,005`).

Partindo do fato de que a quantização pós-treino padrão (`PTQ INT8`) do Edge Impulse já alcançava bom desempenho em 8 bits, o trabalho investiga se o treinamento consciente de quantização (**QAT**) e suas variantes baseadas em gradiente (**QAT Padrão/STE**, **LSQ**, **LSQ+** e **LSQ+ Híbrido**) produzem ganhos em `8` bits (`4x`) e viabilizam compressões extremas **sub-8-bit** em `4` bits (`8x`) e `2` bits (`16x`), sob quantização simulada (*fake quantization*).

---

## Resultados Consolidados (Conjunto de Teste — `252` amostras)

| Modelo | Comp. | Acc (%) | Δ FP32 | Δ PTQ-MSE | Loss | AUC | F1-W |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline FP32** | 1x | 91,27% | 0,00 | — | 0,4191 | 0,9893 | 0,9131 |
| PTQ W8A8 (MINMAX) | 4x | 91,27% | +0,00 pp | +0,40 pp | 0,4183 | 0,9894 | 0,9128 |
| PTQ W8A8 (MSE) | 4x | 90,87% | -0,40 pp | 0,00 pp | 0,4162 | 0,9896 | 0,9090 |
| **QAT Padrão W8A8** | 4x | **91,67%** | +0,40 pp | +0,79 pp | 0,4507 | 0,9884 | 0,9173 |
| QAT LSQ W8A8 | 4x | 91,27% | +0,00 pp | +0,40 pp | 0,4580 | 0,9876 | 0,9130 |
| QAT LSQ+ W8A8 | 4x | 90,48% | -0,79 pp | -0,40 pp | 0,4450 | 0,9883 | 0,9048 |
| QAT LSQ+ Híb W8A8 | 4x | 90,48% | -0,79 pp | -0,40 pp | 0,4556 | 0,9885 | 0,9052 |
| PTQ W4A4 (MINMAX) | 8x | 86,90% | -4,37 pp | -2,38 pp | 0,6890 | 0,9811 | 0,8693 |
| PTQ W4A4 (MSE) | 8x | 89,29% | -1,98 pp | 0,00 pp | 0,4882 | 0,9868 | 0,8929 |
| QAT Padrão W4A4 | 8x | 89,68% | -1,59 pp | +0,40 pp | 0,4421 | 0,9890 | 0,8966 |
| **QAT LSQ W4A4** | 8x | **90,87%** | -0,40 pp | +1,59 pp | 0,5068 | 0,9869 | 0,9094 |
| QAT LSQ+ W4A4 | 8x | 90,08% | -1,19 pp | +0,79 pp | 0,5133 | 0,9851 | 0,9007 |
| QAT LSQ+ Híb W4A4 | 8x | 88,49% | -2,78 pp | -0,79 pp | 0,4764 | 0,9862 | 0,8861 |
| PTQ W2A2 (MINMAX) | 16x | 19,05% | -72,22 pp | -60,32 pp | 11,5572 | 0,6021 | 0,1343 |
| PTQ W2A2 (MSE) | 16x | 79,37% | -11,90 pp | 0,00 pp | 0,8546 | 0,9577 | 0,7955 |
| QAT Padrão W2A2 | 16x | 85,32% | -5,95 pp | +5,95 pp | 0,5197 | 0,9765 | 0,8540 |
| **QAT LSQ W2A2** | 16x | **86,51%** | -4,76 pp | +7,14 pp | **0,4037** | **0,9816** | **0,8663** |
| QAT LSQ+ W2A2 | 16x | 83,33% | -7,94 pp | +3,97 pp | 0,4992 | 0,9736 | 0,8353 |
| QAT LSQ+ Híb W2A2 | 16x | 85,32% | -5,95 pp | +5,95 pp | 0,4703 | 0,9765 | 0,8528 |

---

## Estrutura do Repositório

- `Reconhecimento_Voz_PTQ_vs_QAT.ipynb` — Notebook completo com pré-processamento MFCC, treinamento `FP32`, `PTQ` (`Min-Max` e `MSE`), `QAT Padrão`, `QAT LSQ`, `QAT LSQ+`, ablação `QAT LSQ+ Híbrido`, curvas de convergência e síntese comparativa.
- `baseline_fp32.pth` — Pesos salvos do melhor *checkpoint* do modelo `FP32`.
- `requirements.txt` — Dependências Python para execução local em CPU.

---

## Como Executar

### 1. Localmente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Coloque a pasta do dataset `command_voice_tcc-export` (contendo `training/` e `testing/`) na raiz do repositório ou na pasta pai (`../command_voice_tcc-export`) e execute o notebook `Reconhecimento_Voz_PTQ_vs_QAT.ipynb`.

### 2. No Google Colab
1. Abra diretamente pelo botão **Open in Colab** no topo deste README (ou em **Arquivo → Abrir notebook → GitHub → `pedrohugo-psc/Reconhecimento_Voz_PTQ_vs_QAT`**).
2. Envie a pasta `command_voice_tcc-export` para a raiz do seu **Google Drive** (`MyDrive/command_voice_tcc-export`).
3. Antes de rodar a célula de carregamento do dataset no Colab, adicione e execute uma célula com:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   O notebook detectará automaticamente o dataset em `/content/drive/MyDrive/command_voice_tcc-export`.

---

## Referências

- **[Antoniou & Lu, 2007]** A. Antoniou and W.-S. Lu, *Practical Optimization: Algorithms and Engineering Applications*. Springer, 2007.
- **[Bhalgat et al., 2020]** Y. Bhalgat et al., *"LSQ+: Improving low-bit quantization through learnable offsets and better initialization,"* *CVPRW*, 2020.
- **[Carlos, 2025]** P. H. P. da S. Carlos, *"Sistema Embarcado para Reconhecimento de Palavras usando ESP32-S3 e TinyML,"* TCC (Engenharia de Computação), UFBA, maio 2025.
- **[Lin et al., 2023]** S.-T. Lin et al., *"LG-LSQ: Learned Gradient Linear Symmetric Quantization for Low-Precision Integer Hardware,"* *IEEE AICAS*, 2023.
- **[Pereira et al., 2023]** P. H. Pereira, W. Beccaro, and M. A. Ramírez, *"Evaluating Robustness to Noise and Compression of Deep Neural Networks for Keyword Spotting,"* *IEEE Access*, vol. 11, 2023.
- **[Ravaglia et al., 2021]** L. Ravaglia et al., *"A TinyML Platform for On-Device Continual Learning With Quantized Latent Replays,"* *IEEE JETCAS*, vol. 11, no. 4, Dec. 2021.
- **[Souza, 2026]** E. E. P. de Souza, *Notas de Aula — Otimização (PPGEE0016)*, PPGEEC-UFBA, 2026.
- **[Zhao et al., 2021]** H. Zhao, D. Liu, and H. Li, *"Efficient Integer-Arithmetic-Only Convolutional Networks With Bounded ReLU,"* *IEEE ISCAS*, 2021.
