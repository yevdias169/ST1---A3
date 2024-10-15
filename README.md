# ST1---A3

### **How to Use This Repository**

 This repository contains the Python scripts and datasets we are working on together. Follow the instructions below to set up the project locally, access the CSV file, and contribute changes. 

### **Step 1: Cloning the Repository**

To get started, clone this repository to your local machine using **VS Code**:

1. Open **VS Code**.
2. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) and type `Git: Clone`.
3. Paste the repository link:
   ```bash
   https://github.com/yevdias169/ST1---A3.git
   ```
4. Select a folder to save the project, and VS Code will automatically open the cloned project.

### **Step 2: Accessing the CSV File**

We are using the dataset `Medical_insurance.csv` in this project. The file is located in the root directory of the repository. You can read the file in your Python scripts using the following Pandas command:

```python
import pandas as pd
df = pd.read_csv('Medical_insurance.csv')
```

Make sure not to hardcode any file paths that are specific to your machine. We are all using **relative paths** so the code works universally.

### **Step 4: Making Changes to the Code**

1. Make sure you are working on the **latest version** of the repository:
   ```bash
   git pull
   ```

2. Once you've made your changes:
   - Stage the files by clicking the **plus icon (+)** next to the files under the **Source Control** panel in VS Code.
   - Write a meaningful commit message (e.g., "Updated data processing logic").
   - Commit your changes.

3. Push the changes to the GitHub repository:
   ```bash
   git push
   ```

### **Step 5: Pulling Changes from Other Members**

Before starting work on the project, ensure you have the latest version by pulling the latest changes:

```bash
git pull
```

This will help prevent merge conflicts by syncing your local project with the most recent version from the repository.

### **Step 6: Branching (Optional)**

If you're working on a specific feature or task, it’s a good practice to create a new branch:

1. Create and switch to a new branch:
   ```bash
   git checkout -b my-feature-branch
   ```

2. After completing your work, push the branch and create a pull request (PR) on GitHub for review:
   ```bash
   git push origin my-feature-branch
   ```

---
 If you have any questions, feel free to ask in our group chat!
--- 
Let me know if you’d like any adjustments or more details added to this!