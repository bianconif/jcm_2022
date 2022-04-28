## Description
This repository contains the data and code required to reproduce the results presented in the following paper:

[Bianconi, F.](www.bianconif.net), Conti, P., [Zanetti, E.M.](https://www.linkedin.com/in/elisabetta-zanetti-0a183626/), [Pascoletti, G.](https://www.linkedin.com/in/giulia-pascoletti-273b2618a/) __A benchmark of traditional visual descriptors and convolutional networks 'off-the-shelf' for anomaly detection__, accepted for oral presentation at the International Joint Conference on Mechanics, Design Engineering and Advanced Manufacturing ([JCM 2022](https://www.associazioneadm.it/jcm2022/index.php), June 01-03, 2022 - Ischia, Italy)

## Usage
1. Execute the `experiments.py` script
2. Retrieve the results in LaTeX tabular form in the `output/LaTeX` folder

## Dependencies
- [CenOTAPH v0.1.0-alpha](https://github.com/bianconif/CenOTAPH)
- [Numpy 1.19.1](https://numpy.org/)
- [Pandas 1.0.5](https://pandas.pydata.org/)
- [Python 3.8.6](https://www.python.org/)
- [Tabulate 0.8.9](https://pypi.org/project/tabulate/)

## Datasets
The images used in this study are stored in the `data/images` folder. For each dataset the `./Normal` and `./Abnormal` sub-folders respectively contain the normal (non-defective) and abnormal (defective) samples.

## Credits
- Dataset `Concrete-01` sourced from the [Surface Crack Detection Dataset](https://www.kaggle.com/arunrk7/surface-crack-detection).
- Datasets `Fabric-01` and `Fabric-02` sourced from the [ZJU-Leaper Dataset](https://github.com/nico-zck/ZJU-Leaper-Dataset). Reference paper:
  - Zhang, C., Feng, S., Wang, X., Wang, Y. [ZJU-Leaper} A Benchmark Dataset for Fabric Defect Detection and a Comparative Study](https://doi.org/10.1109/TAI.2021.3057027) (2020) IEEE Transactions on Artificial Intelligence 1 (3), pp. 219-232   
- Dataset `Layered-01` first released here. Reference paper:
  - Rossi, A., Moretti, M., Senin, N. [Layer inspection via digital imaging and machine learning for in-process monitoring of fused filament fabrication](https://doi.org/10.1016/j.jmapro.2021.08.057) (2021) Journal of Manufacturing Processes, 70, pp. 438-451.
- Datasets `Paper-01` and `Paper-02` sourced from [PIPED: Paper ImPuritiEs Dataset](https://github.com/bianconif/PIPED). Reference paper:
  - Bianconi, F., Ceccarelli, L., Fernández, A., Saetta, S.A.; [A sequential machine vision procedure for assessing paper impurities](https://doi.org/10.1016/j.compind.2013.12.001) (2014) Computers in Industry, 65 (2), pp. 325-332.
- Datasets `Carpet-01`, `Leather-01` and `Wood-01` sourced from [MVTEC Anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). Reference papers:
  - Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C. [The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://doi.org/10.1007/s11263-020-01400-4) (2021) International Journal of Computer Vision, 129 (4), pp. 1038-1059.
  - Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. [MVTEC ad-A comprehensive real-world dataset for unsupervised anomaly detection](https://doi.org/10.1109/CVPR.2019.00982) (2019) Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2019-June, art. no. 8954181, pp. 9584-9592.

Please consider citing the sources and the related papers - whenever appropriate - if you wish to use any of the above datasets for your research.

## License
The code in this repository is distributed under [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/); all the other material under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

## Disclaimer
The information and content available on this repository are provided with no warranty whatsoever. Any use for scientific or any other purpose is conducted at your own risk and under your own responsibility. The authors are not liable for any damages - including any consequential damages - of any kind that may result from the use of the materials or information available on this repository or of any of the products or services hereon described.
