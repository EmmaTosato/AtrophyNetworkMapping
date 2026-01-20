# 📝 TO-DO LIST: Nested CV Implementation

## 1. Preparazione
- [x] Analisi Fattibilità (Dynamic Splitting OK)
- [x] Pulizia Documentazione Vecchia
- [ ] Commit e Clean State

## 2. Infrastruttura (Giorno 1)
- [ ] Script `nested_cv_runner.py`
    - [ ] Outer Loop (StratifiedKFold 5-fold)
    - [ ] Inner Loop (Grid Search 5-fold)
    - [ ] Logica Retrain & Test
- [ ] Configurazione `nested_cv_config.json`

## 3. Modelli (Giorno 2)
- [ ] Implementare `AlexNet3D`
- [ ] Aggiornare `ResNet3D` (r3d_18)
- [ ] Verificare `VGG16_3D`

## 4. Verifica e Test (Giorno 3)
- [ ] Dry Run (20 soggetti)
- [ ] Verifica No Data Leakage
- [ ] Verifica Augmentation (Solo Train)

## 5. Esecuzione (Giorno 4)
- [ ] Run ADNI vs PSP
- [ ] Run ADNI vs CBS
- [ ] Run PSP vs CBS
