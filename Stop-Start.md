## **Step1: Picking Up Again the Next Day**

1. Open a terminal.
2. Navigate to your project folder:

```bash
cd ~/tensorflow-projects/my_first_project
```

3. Activate your virtual environment:

```bash
source ~/my-tf-env/bin/activate
```

4. Run your script or TensorBoard:

```bash
python train.py
# or
tensorboard --logdir logs
```


## **Step 2: Closing Everything at the End of the Day**

1. Stop any running scripts or TensorBoard in the terminal:

```
Ctrl + C
```

3. Deactivate your virtual environment:

```bash
deactivate
```