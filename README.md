# AttentionAML: an <ins>Attention</ins>-based MLP Model for Identifying <ins>A</ins>cute <ins>M</ins>yeloid <ins>L</ins>eukemia Subtypes
**AttentionAML** (an **Attention**-Based MLP Model for Identifying **A**cute **M**yeloid **L**eukemia Subtypes), an accurate and effective model for AML subtype identification.

## Flowchart of AttentionAML
![Flowchart of AttentionAML](Flowchart.png)

## Installation
1. Clone the AttentionAML git repository
```bash
git clone https://github.com/wan-mlab/AttentionAML.git
```
2. Navigate to the directory of AttentionAML package
```bash
cd /your path/AttentionAML
pip install .
```
## Tutorial
### Jupyter notebook
1. Modify the System Path and import module
```bash
import sys; sys.path.append('AttentionAML')
from AttentionAML import AttentionAML
```
2. unzip and read the test file
```bash
test = pd.read_csv('TPM_test.csv', index_col=0)
```
3. AML subtype prediction
```bash
AttentionAML.Predict(Exp = test, exp_type = 'TPM')
```
## Authors
Lusheng Li, Shibiao Wan

## Publication
AttentionAML: An attention-based deep learning model for accurate identification of childhood acute myeloid leukemia subtypes
Lusheng Li, Joseph D. Khoury, Jieqiong Wang, Shibiao Wan
Cancer Res (2025) (AACR Abstract)

Manuscript in progress
