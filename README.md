# Tomógrafo de Luz

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreVerasFernandes/Tomografo/blob/main/TOMOGRAFO_Vers%C3%A3o_final.ipynb)

## 📋 Resumo

Um **Tomógrafo** é um equipamento utilizado para a obtenção de imagens detalhadas do interior de objetos. Este projeto apresenta um **Tomógrafo de Luz** que utiliza técnicas de processamento de imagens na linguagem Python para formar a imagem completa de um objeto através da aquisição e tratamento de dados.

O projeto foi desenvolvido como parte do curso de Física Aplicada no SENAC CAS, combinando fundamentos teóricos de Física Elétrica com aplicação prática em programação.

## 👥 Autores

- **Andre Luiz Veras** - duverassusa@gmail.com
- **Andrew Araujo** - andrewaraujoco@gmail.com
- **Gustavo Borges** - guborges4789a@gmail.com
- **Ruan Gomes** - ruanggs.05@gmail.com

**Professores orientadores:**
- Dalke Meucci
- Jorge Echeimberg

## 🎯 Proposta

A ideia do projeto é a produção de um Tomógrafo de Luz, aplicando técnicas de processamento de imagens utilizando Python. O sistema realiza:
- Aquisição de dados através de captura de imagens
- Tratamento e processamento das imagens
- Reconstrução tridimensional do objeto

## 📚 Fundamentos Teóricos

### Transformada de Radon

O processo de tomografia é realizado através de amostragens paralelas de feixes de luz em diferentes ângulos. Esses feixes sofrem atenuações de acordo com a distribuição de densidade do objeto. O conjunto de perfis resultantes é chamado de **Sinograma**, que é a Transformada de Radon do objeto em análise.

A atenuação da intensidade obedece à lei de **Beer-Lambert**. A Transformada de Radon converte uma função de um sistema de coordenadas espaciais (x,y) para outro sistema de coordenadas espaciais de Radon.

### Sinograma

O problema de reconstrução tomográfica consiste em derivar uma imagem tomográfica fatiada a partir de um conjunto de projeções. Cada projeção é formada pela integração do contraste do objeto ao longo de raios paralelos em 2D. A compilação dessas projeções em vários ângulos é denominada sinograma.

## 🔧 Estrutura do Projeto

O dispositivo apresentado é um sistema simplificado de digitalização 3D que incorpora tecnologias inspiradas na tomografia por emissão de luz e scanner 3D utilizando a técnica de "digitalização por projeção de luz estruturada".

### Processo de Captura

- O objeto é dividido em **15 camadas** distintas
- Cada camada é composta por **200 fotos**
- Mecanismos iluminam cada camada enquanto a peça completa uma volta
- A luz é captada por câmeras posicionadas atrás da peça

## 🖥️ Tratamento de Dados

O código está implementado em Python utilizando principalmente as bibliotecas **PIL (Pillow)**, **NumPy**, **Matplotlib** e **scikit-image**.

### 1. Recorte de Imagens

Código para redimensionar e extrair regiões específicas das imagens:

```python
from PIL import Image
import os

def cortar_imagens(pasta_origem, pasta_destino):
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    
    arquivos = os.listdir(pasta_origem)
    
    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_origem, arquivo)
        if os.path.isfile(caminho_completo) and arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            imagem = Image.open(caminho_completo)
            left, top, right, bottom = 1, 190, 399, 203
            imagem_cortada = imagem.crop((left, top, right, bottom))
            caminho_destino = os.path.join(pasta_destino, arquivo)
            imagem_cortada.save(caminho_destino)
```

### 2. Conversão para Escala de Cinza e Inversão

Técnicas para melhorar a visualização do objeto e simplificar o processamento:

```python
from PIL import Image
import os

def converter_tons_de_cinza_e_inverter_densidade(pasta_entrada, pasta_saida):
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    arquivos = os.listdir(pasta_entrada)
    
    for arquivo in arquivos:
        if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_entrada = os.path.join(pasta_entrada, arquivo)
            imagem = Image.open(caminho_entrada)
            imagem_tons_de_cinza = imagem.convert('L')
            imagem_invertida = Image.eval(imagem_tons_de_cinza, lambda x: 255 - x)
            caminho_saida = os.path.join(pasta_saida, arquivo)
            imagem_invertida.save(caminho_saida)
```

### 3. Rotação de Imagens (90 graus)

```python
from PIL import Image
import os

def rotacionar_imagens(diretorio_origem, diretorio_destino):
    if not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino)
    
    arquivos = os.listdir(diretorio_origem)
    
    for arquivo in arquivos:
        if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_imagem_original = os.path.join(diretorio_origem, arquivo)
            imagem = Image.open(caminho_imagem_original)
            imagem_rotacionada = imagem.rotate(90, expand=True)
            caminho_imagem_rotacionada = os.path.join(diretorio_destino, arquivo)
            imagem_rotacionada.save(caminho_imagem_rotacionada)
```

### 4. Geração de Sinogramas

Geração de 15 sinogramas a partir de grupos de 200 imagens:

```python
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

diretorio_imagens = '/caminho/para/imagens'
diretorio_salvar_sinogramas = '/caminho/para/sinogramas/'

intervalos = [(1, 200), (201, 400), (401, 600), (601, 800), (801, 1000),
              (1001, 1200), (1201, 1400), (1401, 1600), (1601, 1800), (1801, 2000),
              (2001, 2200), (2201, 2400), (2401, 2600), (2601, 2800), (2801, 2999)]

for i, (inicio, fim) in enumerate(intervalos, start=1):
    imagens_arrays = []
    for num in range(inicio, fim + 1):
        nome_arquivo = f"{num:05d}.jpg"
        caminho_imagem = os.path.join(diretorio_imagens, nome_arquivo)
        if os.path.exists(caminho_imagem):
            imagem = Image.open(caminho_imagem).convert('L')
            imagens_arrays.append(np.array(imagem))
    
    if imagens_arrays:
        sinograma = np.stack(imagens_arrays, axis=0)
        plt.imshow(sinograma[:, :, 0], cmap='gray')
        plt.title(f"Sinograma {i}")
        plt.savefig(os.path.join(diretorio_salvar_sinogramas, f'sinograma{i}.png'))
        plt.close()
```

### 5. Transformada de Radon e Reconstrução

Aplicação da Transformada de Radon para reconstrução dos planos axiais:

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, iradon
from skimage import io

# Carregar o sinograma gerado na etapa anterior
file_path = '/caminho/para/sinograma.png'
image = io.imread(file_path, as_gray=True)
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)
reconstructed_image = iradon(sinogram, theta=theta, circle=True)

plt.imshow(reconstructed_image, cmap=plt.cm.Greys_r)
plt.title("Imagem Reconstruída")
plt.show()
```

### 6. Visualização 3D

Combinação dos planos axiais para criar uma visualização tridimensional:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread

diretorio_imagens = '/caminho/para/imagens_reconstruidas/'
num_imagens = 15

# Carregar as imagens reconstruídas para formar o volume
vol = []
for i in range(1, num_imagens + 1):
    nome_arquivo = f'radon{i}.png'
    caminho_imagem = os.path.join(diretorio_imagens, nome_arquivo)
    if os.path.exists(caminho_imagem):
        imagem = imread(caminho_imagem, as_gray=True)
        vol.append(imagem)

vol = np.array(vol)

# Criar visualização 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.mgrid[0:vol.shape[1], 0:vol.shape[2]]
for i, slice in enumerate(vol):
    ax.contourf(x, y, slice, zdir='z', offset=i, cmap="gray")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(0, num_imagens)
plt.savefig('visualizacao_3d.png')
plt.show()
```

## 📦 Dependências

```
pillow
numpy
matplotlib
scikit-image
```

## 🚀 Como Usar

1. Clone o repositório:
```bash
git clone https://github.com/AndreVerasFernandes/Tomografo.git
```

2. Instale as dependências:
```bash
pip install pillow numpy matplotlib scikit-image
```

3. Abra o notebook no Google Colab ou Jupyter:
   - Clique no badge "Open In Colab" no topo deste README, ou
   - Execute `jupyter notebook TOMOGRAFO_Versão_final.ipynb`

4. Execute as células sequencialmente para processar suas imagens

## 📊 Resultados

O projeto gera:
- **15 Sinogramas**: Representações das projeções do objeto em diferentes ângulos
- **15 Planos Axiais Reconstruídos**: Imagens reconstruídas usando a Transformada de Radon inversa
- **Visualização 3D**: Combinação dos planos para uma representação tridimensional do objeto

## 🔬 Limitações e Desafios

- Presença de áreas pretas nas bordas das imagens reconstruídas
- Fusão dos planos axiais não completamente consistente
- Necessidade de melhorias na fase de reconstrução 3D
- Complexidade na implementação de algoritmos eficientes para reconstrução tridimensional

## 📖 Referências

- Zhang, H., Wu, J., Zhang, S. (2017). *Light-Emitting Tomography: A Review*. Sensors, 17(3), 560.
- Zhang, S., Zhang, H. (2012). *A review on structured light pattern projection for three-dimensional shape measurement*. Optics and Lasers in Engineering, 50(6), 883-901.
- Salvi, J., Fernandez, S., Pribanic, T., Llado, X. (2010). *A state of the art in structured light patterns for surface profilometry*. Pattern Recognition, 43(8), 2666-2680.
- Araujo, BR (2017). *Reconstrução Tomográfica de imagens SPECT a partir de poucos dados utilizando variação total*. ICMC - USP.
- [Python Pillow Library - Real Python](https://realpython.com/image-processing-with-the-python-pillow-library/)
- [Raspberry Pi Camera Module 2](https://www.raspberrypi.com/products/camera-module-v2/)

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais no SENAC CAS.

---

**Palavras-Chave**: Tomógrafo, Programação, Python, Processamento de Imagens, Transformada de Radon, Reconstrução 3D