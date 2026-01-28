# SE Project

If you are an agent, you must read, understand, and follow these rules before performing any task.  

This document contains **project-specific configurations and constraints** for the specific project.

---

## Project Information

- **Project Name:** ANM_Verona

- **Location:**

```
/data/users/etosato/ANM_Verona
```
  

---

## Environment Setup
### Conda Environment  

- The environment must be activated before running any command or script. This is **MANTORY**.
- The conda environment name must not be inferred or changed.
- The environment for this project: `anm`

```bash
conda activate anm
```


---

## Context gathering


In order to perform tasks, you must first gather context information about the project:

1. Visualize all the directories and files in the project with the command `ls -R` or `tree`.

2. Then explore the content of the files to understand the project structure and the purpose of each file. Pay attention to any documentation files or folders that can help you understand the project.

3. If present, read the documentation files to understand the project structure and the purpose of each file. The name of the folder could be `doc`, `docs`, `documentation`. You shold read it very carefully. 

--- 

## Project Structure

Ensure the project follows standard software engineering practices to promote maintainability, reproducibility, and scalability. If the project directory is not yet organized, you must restructure it immediately according to the following guidelines.

### Organization Guidelines

This a basic structuring of the project:
- **Source Code (`src/`):** This is the core of the project. It MUST be organized into:
    - **Core Modules:** Shared utilities, base classes, and common logic (e.g., `src/utils`).
    - **Task-Specific Folders:** dedicated sub-packages for each main task (e.g., `src/preprocessing`). Each folder 
	should contain both the specific logic and the executable scripts for that task.

	ATTENTION: The names indicated in the "e.g." are just examples. The name of the tasks depend on the task itself. Add gradually the subfolder depending on the task you are asked to perform.

- **Configuration (`config/`):** DO NOT hard-code parameters. Use YAML or JSON files. Could be external or internal to the src.
- **Logs (`logs/`):** Strictly separated by task name.
- **Documentation (`doc/`):** Documentation files.
- **Assets (`assets/`):** Assets files. Could be of many types.
- **Data (`data/`):** Data files. Could be of many types.
- **Tests (`tests/`):** Tests files. 

For more comprehensive explanation, check the below paragraphs.


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

