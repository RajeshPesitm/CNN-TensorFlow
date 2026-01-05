# Setup – One-Time Initial Setup

This guide will help you understand how **original author** prepared the system to run TensorFlow projects on Ubuntu 22.04.

## **Step 1: Check & Install Python, Pip, TensorFlow**

1. **Check Python 3 version:**

```bash
python3 --version
```

* You should see something like `Python 3.10.x`.

2. **Check pip version:**

```bash
pip3 --version
```

* If pip is missing, install it:

```bash
sudo apt update
sudo apt install python3-pip -y
```

3. **Install virtual environment support (recommended):**
### **Try creating a virtual environment**

Run this in your terminal:

```bash
python3 -m venv ~/my-tf-env
```

* **If it fails:**

```bash
sudo apt install python3-venv -y
```
*Try Again: Your terminal prompt should now show `(my-tf-env)`.*


4. **Create a virtual environment for your project:**

```bash
python3 -m venv ~/my-tf-env
```

5. **Activate the virtual environment:**

```bash
source ~/my-tf-env/bin/activate
```

* Your prompt should now show `(my-tf-env)`.

6. **Install TensorFlow (CPU version):**

```bash
pip install --upgrade pip
pip install tensorflow
```

7. **Optional: Install extra tools you might need** (like OpenCV or tqdm):

```bash
pip install opencv-python tqdm matplotlib
```

8. **Verify TensorFlow installation:**

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## **Step 2: Create & Run a TensorFlow Project**

1. **Create a project folder:**

```bash
mkdir ~/tensorflow-projects
cd ~/tensorflow-projects
mkdir my_first_project
cd my_first_project
```

2. **Create your Python script**, e.g., `train.py`:

```bash
nano train.py
```

* Paste your TensorFlow/Keras code in here.
* Save with `Ctrl+O` → Enter → `Ctrl+X` to exit.

3. **Run the script:**

```bash
python train.py
```

4. **Optional: Run TensorBoard** (if you’re using it):

```bash
tensorboard --logdir logs
```

* Open a browser at `http://localhost:6006` to see your training graphs.

---

## **Step 3: Freez dependensies to requirements.txt**

```bash
pip freeze > requirements.txt
```

## **Step 5: Add Appropriate .gitignore file**



## **Step 6: Closing Everything at the End of the Day**

1. Stop any running scripts or TensorBoard in the terminal:

```
Ctrl + C
```

3. Deactivate your virtual environment:

```bash
deactivate
```