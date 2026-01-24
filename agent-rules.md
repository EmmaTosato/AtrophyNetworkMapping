# SE Project

If you are an agent, you must read, understand, and follow these rules before performing any task.  

This document contains **project-specific configurations and constraints** for the specific project.

---

## Project Information

- **Project Name:** Cebra

- **Location:**

```
/data/users/etosato/ANM_Verona
```
  

---

## Environment Setup
### Conda Environment  

- The environment must be activated before running any command or script. This is **MANTORY**.
- The environment for this project: `cebra`

```bash
conda activate anm
```

* The conda environment name must not be inferred or changed.

---

## Context gathering

  

In order to perform tasks, you must first gather context information about the project:

1. Visualize all the directories and files in the project with the command `ls -R` or `tree`.

2. Then explore the content of the files to understand the project structure and the purpose of each file.

3. Pay attention to any documentation files or folders that can help you understand the project.

--- 

## Project Structure

Ensure the project follows standard software engineering practices to promote maintainability, reproducibility, and scalability. If the project directory is not yet organized, you must restructure it immediately according to the following guidelines.

### Organization Guidelines

- **Source Code & Scripts (`src/`):** This is the core of the project. It MUST be organized into:
    - **Core Modules:** Shared utilities, base classes, and common logic (e.g., `src/utils`, `src/models`).
    - **Task-Specific Folders:** dedicated sub-packages for each main task (e.g., `src/preprocessing`, `src/training`). Each folder should contain both the specific logic and the executable scripts for that task.
- **Configuration (`config/`):** DO NOT hard-code parameters. Use YAML or JSON files.
- **Logs (`logs/`):** Strictly separated by task name.
- **SLURM Jobs (`jobs/`):** (Optional) If you prefer keeping `.sh` submission files separate from Python code, place them here, mirroring the `src` task structure.

### Standard Hierarchy Example

The project file system could be structured as follows. It doesn't have to be exactly like this, the point is to maintain a standard and clean organization.

```
/data/etosato/cebra/ # Project Root
│
├── config/ # Configuration files
│ ├── main_config.yaml
│ └── model_params.json
│
├── data/ # Data storage
│ ├── raw/
│ └── processed/
│
├── doc/ # Documentation
│ └── setup.md
│
├── logs/ # Centralized logs
│ └── <task_name>/ # Task-specific subfolders
│ └── <job_name>_<id>.out
│
├── src/ # Source Code & Scripts
│ ├── __init__.py
│ │
│ ├── core/ # Reusable shared code
│ │ ├── loaders.py
│ │ └── utils.py
│ │
│ └── tasks/ # Organized by functional task
│ ├── preprocessing/ # Task 1: Data Preparation
│ │ ├── clean.py
│ │ └── normalize.py
│ │
│ └── analysis/ # Task 2: Analysis/Training
│ ├── train_cebra.py
│ └── decoding.py
│
├── jobs/ # (Optional) SLURM .sh submission scripts
│ └── analysis/
│ └── submit_train.sh
│
...
...
├── .gitignore
├── environment.yml
└── README.md
```

### Logs

- A project MUST have a logs folder. If it doesn't exist, create it.
- For each main task, create a log subfolder in the main logs folder.
- All the log files related to that task MUST be stored in the task's log subfolder.
- The log files inside the task's log subfolder MUST be differentiable, so pay attention to the name of the log files.

Doc
- If not already present, create a `doc` folder. Here, the agent must add the documentation created incrementally with the realization of the project 
- If already present, the agent should continue to modify and update the documentation.
- What to put?
	- Explanation of the project
	- User guide
	- What the user ask to the agent to add/explain

--- 
## Language Rules

* **Code and documentation:** English only
* **Chat language:** If the user writes in Italian, respond in Italian
- Italian is allowed **only in the chat**, never in:
	- source code
	- comments
	- documentation
	- configuration files
- You **MUST NOT** use emoticons. Especially
	- in the code
	- in the documentation

---

  
Failure to follow these rules is considered a violation of the project constraints.

