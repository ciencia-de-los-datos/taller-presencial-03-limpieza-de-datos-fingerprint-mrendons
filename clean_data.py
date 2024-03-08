"""Taller evaluable presencial"""

import nltk
import pandas as pd
import string

def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""
    df = pd.read_csv(input_file)
    return df




def create_fingerprint(df):
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""

    # 1. Copie la columna 'text' a la columna 'fingerprint'
    # 2. Remueva los espacios en blanco al principio y al final de la cadena
    # 3. Convierta el texto a minúsculas
    # 4. Transforme palabras que pueden (o no) contener guiones por su version sin guion.
    # 5. Remueva puntuación y caracteres de control
    # 6. Convierta el texto a una lista de tokens
    # 7. Transforme cada palabra con un stemmer de Porter
    # 8. Ordene la lista de tokens y remueve duplicados
    # 9. Convierta la lista de tokens a una cadena de texto separada por espacios

    df = df.copy() #obligatoria, para no modificar el real
    df["key"] = df["text"] #nueva columna con el contenido de la columna text
    df["key"] = df["key"].str.strip() #paso 2
    df["key"] = df["key"].str.lower() #paso 3
    df["key"] = df["key"].str.replace("-","") #paso 4
    df["key"] = df["key"].str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    ) #paso 5
    df["key"] = df["key"].str.split()#paso 6 lo convierte en una lista con n elementos (palabras)
    stemmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(lambda x:[stemmer.stem(word)for word in x]) #paso 7 transforma las palabras para generar la raiz de la palabra
    #a la columna key le aplica a cada elemento de la columna la función lambda cuyo variable es x, a cada palabra apliquele el stemmer 
    df["key"] = df["key"].apply(lambda x:sorted(set(x))) #paso 8 ordena con sorted y set convierte la lista en un conjunto de datos ordenados sin elementos repetidos
    df["key"] = df["key"].str.join(" ") #paso 9 
    
    return df


def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()
    df = df.sort_values(by=["key", "text"], ascending=[True, True])
    keys = df.drop_duplicates(subset="key",keep="first")
    key_dict = dict(zip(keys["key"],keys["text"]))#zip aparea. 
    df["cleaned"] = df["key"].map(key_dict) #Si se mapea un diccionario en una columna, le aplica el diccionario al elemento
    return df



    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    # 2. Seleccione la primera fila de cada grupo de 'fingerprint'
    # 3.  Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    # 4. Cree la columna 'cleaned' usando el diccionario


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    
    df = df.copy() #IMPORTANTE
    df = df[["cleaned"]]
    df = df.rename(columns={"cleaned": "text"})
    df.to_csv(output_file, index=False)

def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
