## Create environment (within group5 folder)

```

python -m venv ./virtualenv

```

## Activate environment

Venv on Mac:
```

source virtualenv/bin/activate

```

Venv on Win:
```

source virtualenv/Scripts/activate

```

## Install package
```

python -m pip install <package_name>

```

## Install specific package version
```

python -m pip install requests==2.6.0

```

## Upgrade package
```

python -m pip install --upgrade requests

```

## Export list of dependencies
```

pip freeze > requirements.txt

```

## Install dependencies
```

python -m pip install -r requirements.txt

```