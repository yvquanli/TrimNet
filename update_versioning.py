import re

# Obtém a versão atual do arquivo setup.py
with open('setup.py', 'r') as f:
    setup_content = f.read()
    version_match = re.search(
        r"version=['\"](\d+\.\d+\.\d+)['\"]", setup_content)
    if version_match is None:
        print("Erro: Não foi possível encontrar a versão no arquivo setup.py")
        exit(1)
    current_version = version_match.group(1)

# Gera uma nova versão
new_version = current_version[0:-1] + str(int(current_version[-1]) + 1) 

# Atualiza o arquivo setup.py com a nova versão
new_setup_content = re.sub(
    r"version=['\"]\d+\.\d+\.\d+['\"]", "version='{}'".format(new_version),
    setup_content)
with open('setup.py', 'w') as f:
    f.write(new_setup_content)