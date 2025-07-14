### Download PushT Data
Under the repo root, create data subdirectory:
```console
mkdir data && cd data
```

Download the zip file from [https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/)
```console
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
```

Extract data:
```console
unzip pusht.zip && rm -f pusht.zip && cd ..
```