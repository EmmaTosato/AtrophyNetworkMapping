# 📝 TO-DO LIST: Nested CV Implementation

## 1. Preparazione
- [x] Analisi Fattibilità (Dynamic Splitting OK) - Fatto
- [x] Pulizia Documentazione Vecchia - Fatto
- [x] Commit e Clean State

## 2. Infrastruttura
- [x] Script `nested_cv_runner.py`
    - [x] Outer Loop (StratifiedKFold 5-fold)
    - [x] Inner Loop (Grid Search 5-fold)
    - [x] Logica Retrain & Test
- [x] Configurazione JSON per iperparametri

## 3. Modelli
- [x] Implementazione `AlexNet3D`
- [x] Aggiornamento `ResNet3D` (r3d_18)
- [x] Verifica `VGG16_3D`

## 4. Verifica
- [x] Dry Run (20 soggetti)
- [x] Verifica No Data Leakage
- [x] Verifica Augmentation (Solo Train)

## 5. Esecuzione
- [ ] Calcolo dei tempi facendo un one shot training con 40 epoche per capire quanto ci vuole
- [/] **Run AD vs PSP** (Running - ResNet)
- [/] **Run AD vs CBS** (Running - ResNet)
- [/] **Run PSP vs CBS** (Running - ResNet)

## 6. Future Improvements (Idee)
- [ ] **Run Management System**:
    - Usare nomi sequenziali: `run1`, `run2`, `run3`...
    - Mantenere un file indice (`runs_description.md` o JSON) che mappa:
      `run1` -> "Test iniziale LR 0.01"
      `run2` -> "Prova con nuovo scheduler"
    - Evitare timestamp complessi nei nomi cartella.
