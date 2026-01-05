# Users – Cloning and Running Projects from Others

This guide is for **other users** who want to use or contribute to the TensorFlow project.

## **Local System Requirements: Python, Pip, TensorFlow**

Before running any Python or TensorFlow projects, ensure your system meets the following requirements:

### 1. **Python 3**

* Check your Python version:

```bash
python3 --version
```

* Required version: **Python 3.10.x** or higher.

* If Python is not installed, install it using your system’s package manager (e.g., `apt` for Ubuntu):

```bash
sudo apt update
sudo apt install python3 -y
```

---

### 2. **Pip (Python Package Installer)**

* Check if pip is installed:

```bash
pip3 --version
```

* If pip is missing, install it:

```bash
sudo apt update
sudo apt install python3-pip -y
```

* If pip is installed already, its a good practice to upgrade:

```bash
 pip3 install --upgrade pip3
```

---



# Clone and Run

## Step 1: Clone the Repository

```bash
git clone https://github.com/username/repo_name.git
cd repo_name
```

## Step 2: Create and Activate Virtual Environment

1. Create a virtual environment inside the cloned repo:

```bash
python3 -m venv my-tf-env
```

2. Activate it:

```bash
source my-tf-env/bin/activate
```

* **If it fails:**

```bash
sudo apt install python3-venv -y
```
*Try Activating Again: Your terminal prompt should now show `(my-tf-env)`.*

---

## Step 3: Install Project Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

*This ensures your environment matches the original author's setup.*

---

## Step 4: Run the Project

Run your main script or TensorBoard from the project folder:

```bash
python3 train.py
# or
tensorboard --logdir logs
```

*No need to `cd` again if you are already inside the cloned repo.*

---
