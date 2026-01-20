# 📝 TO-DO LIST: Nested CV Implementation

## 1. Preparazione
- [x] Analisi Fattibilità (Dynamic Splitting OK) - Fatto
- [x] Pulizia Documentazione Vecchia - Fatto
- [ ] Commit e Clean State

## 2. Infrastruttura
- [ ] Script `nested_cv_runner.py`
    - [ ] Outer Loop (StratifiedKFold 5-fold)
    - [ ] Inner Loop (Grid Search 5-fold)
    - [ ] Logica Retrain & Test
- [ ] Configurazione JSON per iperparametri

## 3. Modelli
- [ ] Implementazione `AlexNet3D`
- [ ] Aggiornamento `ResNet3D` (r3d_18)
- [ ] Verifica `VGG16_3D`

## 4. Verifica
- [ ] Dry Run (20 soggetti)
- [ ] Verifica No Data Leakage
- [ ] Verifica Augmentation (Solo Train)

## 5. Esecuzione
- [ ] Run ADNI vs PSP
- [ ] Run ADNI vs CBS
- [ ] Run PSP vs CBS
