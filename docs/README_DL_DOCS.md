# 📚 Deep Learning Pipeline Documentation

Documentazione completa per la pipeline Deep Learning per classificazione di malattie neurodegenerative.

---

## 📖 Guide Disponibili

### 🎯 [DL_PIPELINE_OVERVIEW.md](./DL_PIPELINE_OVERVIEW.md)
**Per chi**: Tutti (overview generale)  
**Contenuto**:
- Diagramma workflow completo (Tuning → Runs → Testing)
- Filosofia della pipeline (perché CV, perché seed multipli)
- Struttura dati e output
- Quick start commands

**Leggi se**: Vuoi capire il quadro generale della pipeline.

---

### 🔬 [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md)
**Per chi**: Sviluppatori, AI agents, ricercatori tecnici  
**Contenuto**:
- Architettura dettagliata del codice
- Function reference completa
- Data flow interno
- Meccaniche di training (seed, CV, early stopping)
- Checkpoint structure
- Performance benchmarks

**Leggi se**: Devi modificare il codice, debuggare, o capire i dettagli implementativi.

---

### 🚀 [DL_USER_GUIDE.md](./DL_USER_GUIDE.md)
**Per chi**: Utenti finali (studenti, ricercatori)  
**Contenuto**:
- Step-by-step da tuning a testing
- Configurazione file JSON
- Come interpretare i risultati
- Troubleshooting comuni
- Best practices

**Leggi se**: Devi **usare** la pipeline per i tuoi esperimenti.

---

## 🗂️ Quick Navigation

### Voglio...

#### ...capire come funziona la pipeline
→ Leggi: [DL_PIPELINE_OVERVIEW.md](./DL_PIPELINE_OVERVIEW.md)

#### ...lanciare un esperimento
→ Leggi: [DL_USER_GUIDE.md](./DL_USER_GUIDE.md)

#### ...modificare il codice
→ Leggi: [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md)

#### ...interpretare i risultati
→ Leggi: [DL_USER_GUIDE.md](./DL_USER_GUIDE.md) - Sezione "Analizza i risultati"

#### ...debuggare un errore
→ Leggi: [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) - Sezione "Debugging & Logs"

#### ...aggiungere un nuovo modello
→ Leggi: [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) - Sezione "Models"

---

## 📊 Struttura del Progetto

```
ANM_Verona/
├── src/DL_analysis/
│   ├── cnn/                    # Modelli e dataset
│   ├── training/               # Training e tuning
│   ├── testing/                # Evaluation
│   ├── config/                 # Configurazioni JSON
│   └── utils/                  # Utility functions
│
├── data/
│   ├── FCmaps_processed/       # Dati originali
│   └── FCmaps_augmented_processed/  # Dati augmented
│
├── assets/split_cnn/           # Split fissi train/test
│
├── results/
│   ├── tuning/                 # Hyperparameter search results
│   └── runs/                   # Final runs results
│
└── docs/                       # QUESTA DOCUMENTAZIONE
    ├── README_DL_DOCS.md       # Questo file
    ├── DL_PIPELINE_OVERVIEW.md
    ├── DL_TECHNICAL_REFERENCE.md
    └── DL_USER_GUIDE.md
```

---

## 🎓 Percorsi di Lettura Consigliati

### Per Nuovi Utenti
1. Leggi [DL_PIPELINE_OVERVIEW.md](./DL_PIPELINE_OVERVIEW.md) per il big picture
2. Segui [DL_USER_GUIDE.md](./DL_USER_GUIDE.md) step-by-step
3. Consulta [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) se serve approfondire

### Per Sviluppatori
1. Leggi [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) per l'architettura
2. Consulta [DL_PIPELINE_OVERVIEW.md](./DL_PIPELINE_OVERVIEW.md) per la filosofia
3. Usa [DL_USER_GUIDE.md](./DL_USER_GUIDE.md) per verificare che le modifiche funzionino

### Per AI Agents
1. Leggi [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) per dettagli implementativi
2. Riferisciti a [DL_PIPELINE_OVERVIEW.md](./DL_PIPELINE_OVERVIEW.md) per workflow e output
3. Usa [DL_USER_GUIDE.md](./DL_USER_GUIDE.md) per comandi pratici

---

## 🔗 Links Utili

### File Principali
- **Main Runner**: `src/DL_analysis/training/run.py`
- **Tuning**: `src/DL_analysis/training/hyper_tuning.py`
- **Config Base**: `src/DL_analysis/config/cnn_config.json`
- **Grid Search**: `src/DL_analysis/config/cnn_grid.json`

### Output Importanti
- **Training Results**: `results/runs/all_training_results.csv`
- **Testing Results**: `results/runs/all_testing_results.csv`
- **Tuning Results**: `results/tuning/tuning{N}/grid_results.csv`

### Log Files
- **Training Log**: `results/runs/run{N}/log_train{N}`
- **Testing Log**: `results/runs/run{N}/log_test{N}`

---

## 🆘 Troubleshooting Quick Links

| Problema | Guida | Sezione |
|----------|-------|---------|
| Out of Memory | [DL_USER_GUIDE.md](./DL_USER_GUIDE.md) | Troubleshooting |
| Low test accuracy | [DL_PIPELINE_OVERVIEW.md](./DL_PIPELINE_OVERVIEW.md) | Philosophy |
| Seed management | [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) | Seed Management |
| Different results | [DL_USER_GUIDE.md](./DL_USER_GUIDE.md) | Troubleshooting |
| Config file errors | [DL_TECHNICAL_REFERENCE.md](./DL_TECHNICAL_REFERENCE.md) | Configuration System |

---

## 📈 Versioning

- **v1.0** (Gennaio 2026): Documentazione iniziale completa
  - DL_PIPELINE_OVERVIEW.md
  - DL_TECHNICAL_REFERENCE.md  
  - DL_USER_GUIDE.md

---

## 🤝 Contributi

Per aggiornare la documentazione:
1. Modifica il file appropriato in `docs/`
2. Aggiorna questo README se aggiungi nuove sezioni
3. Mantieni la struttura consistente (emoji, markdown, esempi)

---

**Ultima modifica**: Gennaio 2026  
**Autore**: Pipeline DL per classificazione neurodegenerativa
