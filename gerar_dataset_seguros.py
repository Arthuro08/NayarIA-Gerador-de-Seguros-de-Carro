import numpy as np
import pandas as pd

def gerar_dataset(n_linhas=100000, seed=42):
    np.random.seed(seed)
    
    # 1. Idade: entre 18 e 78 anos
    idades = np.random.normal(loc=38, scale=12, size=n_linhas).astype(int)
    idades = np.clip(idades, 18, 78)
    
    # 2. Salário mensal: lognormal realista (R$ 1.600 a R$ 90.000)
    fator_idade_salario = 1 + np.clip((idades - 18) / 40.0, 0, 1.2) * 0.5
    salario_base = np.random.lognormal(mean=8.6, sigma=0.58, size=n_linhas)
    salarios = np.clip(salario_base * fator_idade_salario, 1600.0, 95000.0)
    
    # 3. Catálogo abrangente: 27 Marcas e mais de 150 modelos
    # Cada tupla contém: (Nome do Modelo, Preço Base 0km aproximado, Desvio Padrão)
    catalogo = {
        'Chevrolet': [
            ('Onix', 86000, 7000),
            ('Onix Plus', 94000, 8000),
            ('Tracker', 132000, 11000),
            ('Cruze', 148000, 12000),
            ('Spin', 108000, 9000),
            ('S10', 235000, 25000),
            ('Montana', 126000, 9000),
            ('Equinox', 215000, 18000),
            ('Celta', 38000, 4000),
            ('Corsa', 42000, 5000),
            ('Astra', 48000, 5000),
            ('Prisma', 68000, 6000),
            ('Camaro', 490000, 40000),
        ],
        'Fiat': [
            ('Mobi', 73000, 5000),
            ('Uno', 45000, 5000),
            ('Palio', 42000, 5000),
            ('Siena', 50000, 6000),
            ('Punto', 55000, 6000),
            ('Argo', 88000, 7000),
            ('Cronos', 94000, 8000),
            ('Pulse', 114000, 9000),
            ('Fastback', 138000, 11000),
            ('Toro', 158000, 15000),
            ('Strada', 116000, 10000),
            ('Fiorino', 105000, 8000),
            ('Titano', 230000, 20000),
        ],
        'Volkswagen': [
            ('Gol', 68000, 6000),
            ('Fox', 54000, 6000),
            ('Up', 58000, 6000),
            ('Voyage', 65000, 7000),
            ('Polo', 94000, 8000),
            ('Virtus', 120000, 9000),
            ('Nivus', 138000, 10000),
            ('T-Cross', 150000, 12000),
            ('Taos', 198000, 15000),
            ('Tiguan', 275000, 22000),
            ('Jetta', 225000, 18000),
            ('Golf', 125000, 15000),
            ('Amarok', 290000, 28000),
            ('Saveiro', 98000, 8000),
        ],
        'Hyundai': [
            ('HB20', 89000, 7000),
            ('HB20S', 97000, 8000),
            ('Creta', 140000, 12000),
            ('i30', 65000, 8000),
            ('ix35', 95000, 10000),
            ('Tucson', 185000, 16000),
            ('Santa Fe', 295000, 25000),
            ('Azera', 110000, 15000),
            ('Kona', 190000, 15000),
        ],
        'Toyota': [
            ('Etios', 65000, 6000),
            ('Yaris', 104000, 8000),
            ('Yaris Sedan', 110000, 8000),
            ('Corolla', 165000, 14000),
            ('Corolla Cross', 185000, 15000),
            ('Hilux', 280000, 25000),
            ('SW4', 385000, 30000),
            ('RAV4', 320000, 25000),
            ('Camry', 340000, 30000),
        ],
        'Honda': [
            ('Fit', 75000, 7000),
            ('City', 120000, 9000),
            ('City Hatchback', 122000, 9000),
            ('Civic', 175000, 16000),
            ('HR-V', 165000, 14000),
            ('WR-V', 98000, 8000),
            ('ZR-V', 215000, 16000),
            ('CR-V', 265000, 22000),
            ('Accord', 310000, 25000),
        ],
        'Renault': [
            ('Kwid', 73000, 5000),
            ('Clio', 35000, 4000),
            ('Sandero', 79000, 6000),
            ('Logan', 84000, 6000),
            ('Duster', 124000, 10000),
            ('Captur', 115000, 10000),
            ('Oroch', 128000, 9000),
            ('Kardian', 118000, 8000),
            ('Fluence', 62000, 7000),
        ],
        'Jeep': [
            ('Renegade', 140000, 12000),
            ('Compass', 190000, 16000),
            ('Commander', 248000, 20000),
            ('Cherokee', 135000, 18000),
            ('Grand Cherokee', 380000, 35000),
            ('Wrangler', 440000, 35000),
        ],
        'Nissan': [
            ('March', 56000, 6000),
            ('Versa', 108000, 8000),
            ('Sentra', 158000, 13000),
            ('Kicks', 130000, 10000),
            ('Frontier', 258000, 22000),
            ('Tiida', 42000, 5000),
        ],
        'Ford': [
            ('Ka', 58000, 6000),
            ('Ka Sedan', 62000, 6000),
            ('Fiesta', 52000, 6000),
            ('Focus', 75000, 9000),
            ('EcoSport', 82000, 9000),
            ('Ranger', 270000, 25000),
            ('Maverick', 225000, 18000),
            ('Territory', 210000, 16000),
            ('Bronco Sport', 255000, 20000),
            ('Fusion', 110000, 15000),
            ('Mustang', 540000, 45000),
        ],
        'Peugeot': [
            ('206', 28000, 4000),
            ('207', 34000, 4000),
            ('208', 92000, 8000),
            ('2008', 125000, 11000),
            ('3008', 240000, 20000),
            ('308', 68000, 8000),
        ],
        'Citroen': [
            ('C3', 82000, 7000),
            ('C3 Aircross', 118000, 9000),
            ('C4 Cactus', 115000, 9000),
            ('C4 Lounge', 70000, 8000),
            ('Basalt', 105000, 8000),
            ('Aircross', 65000, 7000),
        ],
        'Mitsubishi': [
            ('Lancer', 78000, 9000),
            ('ASX', 95000, 10000),
            ('Eclipse Cross', 175000, 14000),
            ('Outlander', 230000, 20000),
            ('Pajero TR4', 68000, 8000),
            ('Pajero Sport', 370000, 30000),
            ('L200 Triton', 265000, 24000),
        ],
        'Kia': [
            ('Picanto', 55000, 6000),
            ('Cerato', 115000, 12000),
            ('Soul', 72000, 8000),
            ('Sportage', 235000, 20000),
            ('Sorento', 285000, 25000),
            ('Stonic', 138000, 10000),
            ('Niro', 205000, 15000),
        ],
        'CAOA Chery': [
            ('Tiggo 2', 78000, 7000),
            ('Tiggo 5X', 132000, 10000),
            ('Tiggo 7 Pro', 175000, 14000),
            ('Tiggo 8', 215000, 18000),
            ('Arrizo 6', 135000, 12000),
            ('iCar', 118000, 9000),
        ],
        'BYD': [
            ('Dolphin Mini', 118000, 8000),
            ('Dolphin', 152000, 10000),
            ('Yuan Plus', 232000, 15000),
            ('Song Plus', 242000, 16000),
            ('Seal', 298000, 20000),
            ('King', 178000, 12000),
        ],
        'GWM': [
            ('Ora 03', 155000, 10000),
            ('Haval H6', 220000, 16000),
            ('Haval H6 GT', 315000, 22000),
            ('Tank 300', 360000, 25000),
        ],
        'BMW': [
            ('118i', 265000, 20000),
            ('320i', 340000, 25000),
            ('330e', 410000, 30000),
            ('530i', 480000, 35000),
            ('X1', 320000, 22000),
            ('X3', 435000, 35000),
            ('X5', 680000, 50000),
            ('M3', 790000, 60000),
            ('Z4', 490000, 40000),
        ],
        'Mercedes-Benz': [
            ('A200', 298000, 22000),
            ('CLA 200', 335000, 25000),
            ('C180', 340000, 25000),
            ('C200', 380000, 28000),
            ('C300', 440000, 32000),
            ('GLA 200', 350000, 25000),
            ('GLB 200', 385000, 28000),
            ('GLC 300', 495000, 38000),
            ('G63', 1850000, 120000),
        ],
        'Audi': [
            ('A1', 95000, 10000),
            ('A3 Sedan', 288000, 22000),
            ('A4', 340000, 25000),
            ('A5', 395000, 30000),
            ('Q3', 315000, 24000),
            ('Q5', 430000, 32000),
            ('Q7', 620000, 45000),
            ('e-tron', 590000, 40000),
            ('RS3', 580000, 45000),
        ],
        'Volvo': [
            ('EX30', 230000, 16000),
            ('XC40', 320000, 22000),
            ('XC60', 430000, 30000),
            ('XC90', 580000, 40000),
            ('C40', 350000, 25000),
            ('V40', 110000, 12000),
        ],
        'Land Rover': [
            ('Discovery Sport', 380000, 28000),
            ('Discovery', 690000, 50000),
            ('Range Rover Evoque', 440000, 32000),
            ('Range Rover Velar', 610000, 45000),
            ('Defender', 720000, 55000),
            ('Range Rover Sport', 890000, 70000),
        ],
        'Porsche': [
            ('Macan', 520000, 40000),
            ('Cayenne', 780000, 60000),
            ('Panamera', 890000, 70000),
            ('718 Boxster', 620000, 45000),
            ('718 Cayman', 610000, 45000),
            ('911 Carrera', 980000, 80000),
            ('Taycan', 760000, 60000),
        ],
        'Subaru': [
            ('Impreza', 85000, 9000),
            ('XV', 135000, 11000),
            ('Forester', 225000, 16000),
            ('Outback', 265000, 20000),
            ('WRX', 370000, 30000),
        ],
        'Suzuki': [
            ('Jimny', 95000, 9000),
            ('Jimny Sierra', 158000, 12000),
            ('Vitara', 130000, 10000),
            ('Grand Vitara', 82000, 8000),
            ('Swift', 52000, 6000),
        ],
        'Mini': [
            ('Cooper', 225000, 16000),
            ('Cooper S', 265000, 20000),
            ('Countryman', 295000, 22000),
        ],
        'Lexus': [
            ('UX 250h', 290000, 20000),
            ('NX 350h', 390000, 28000),
            ('RX 450h', 520000, 38000),
            ('ES 300h', 380000, 28000),
        ]
    }
    
    # Categorias de marcas para probabilidades e regras de negócio
    marcas_populares = ['Chevrolet', 'Fiat', 'Volkswagen', 'Hyundai', 'Renault', 'Ford', 'Peugeot', 'Citroen']
    marcas_medias = ['Toyota', 'Honda', 'Jeep', 'Nissan', 'Mitsubishi', 'Kia', 'CAOA Chery', 'BYD', 'GWM', 'Suzuki', 'Subaru']
    marcas_premium = ['BMW', 'Mercedes-Benz', 'Audi', 'Volvo', 'Mini', 'Lexus']
    marcas_super_luxo = ['Porsche', 'Land Rover']
    
    todas_marcas = list(catalogo.keys())
    
    # Pesos de marcas conforme perfil de renda
    pesos_base = {
        'Chevrolet': 0.12, 'Fiat': 0.12, 'Volkswagen': 0.12, 'Hyundai': 0.09,
        'Toyota': 0.08, 'Honda': 0.07, 'Renault': 0.06, 'Jeep': 0.05,
        'Ford': 0.05, 'Nissan': 0.04, 'BYD': 0.035, 'CAOA Chery': 0.025,
        'Peugeot': 0.025, 'Citroen': 0.025, 'Mitsubishi': 0.02, 'GWM': 0.02,
        'BMW': 0.015, 'Mercedes-Benz': 0.015, 'Audi': 0.012, 'Kia': 0.012,
        'Volvo': 0.008, 'Mini': 0.005, 'Suzuki': 0.005, 'Subaru': 0.004,
        'Land Rover': 0.003, 'Porsche': 0.003, 'Lexus': 0.002
    }
    
    # 4. Anos de fabricação (2005 a 2025)
    pesos_anos = np.linspace(1, 9, 2025 - 2005 + 1)
    pesos_anos /= pesos_anos.sum()
    anos_possiveis = np.arange(2005, 2026)
    anos = np.random.choice(anos_possiveis, size=n_linhas, p=pesos_anos)
    
    # Pré-computa vetores de probabilidade para ganho expressivo de performance
    p_dict_baixa = {m: (0.10 if m in marcas_populares else 0.015 if m in marcas_medias else 0.0) for m in todas_marcas}
    p_baixa = np.array([p_dict_baixa.get(m, 0.01) for m in todas_marcas])
    p_baixa /= p_baixa.sum()

    p_dict_alta = {m: (0.015 if m in marcas_populares else 0.035 if m in marcas_medias else 0.08 if m in marcas_premium else 0.12) for m in todas_marcas}
    p_alta = np.array([p_dict_alta.get(m, 0.01) for m in todas_marcas])
    p_alta /= p_alta.sum()

    p_dict_media_alta = {m: (0.025 if m in marcas_populares else 0.055 if m in marcas_medias else 0.05 if m in marcas_premium else 0.01) for m in todas_marcas}
    p_media_alta = np.array([p_dict_media_alta.get(m, 0.01) for m in todas_marcas])
    p_media_alta /= p_media_alta.sum()

    p_base = np.array([pesos_base.get(m, 0.01) for m in todas_marcas])
    p_base /= p_base.sum()

    marcas_escolhidas = []
    nomes_escolhidos = []
    valores_carro = []
    
    for i in range(n_linhas):
        sal = salarios[i]
        ano = anos[i]
        
        # Ajuste dinâmico de probabilidade por renda
        if sal < 4500:
            p_valores = p_baixa
        elif sal > 28000:
            p_valores = p_alta
        elif sal > 12000:
            p_valores = p_media_alta
        else:
            p_valores = p_base
            
        marca = np.random.choice(todas_marcas, p=p_valores)
        
        # Sorteia modelo dentro da marca
        modelos = catalogo[marca]
        idx_mod = np.random.randint(0, len(modelos))
        nome_carro, media_preco, desv_preco = modelos[idx_mod]
        
        # Preço base 0km
        valor_zero = max(25000.0, np.random.normal(media_preco, desv_preco))
        
        # Depreciação anual composta (6.5% a 9.5% ao ano)
        anos_uso = 2025 - ano
        taxa_depr = np.random.uniform(0.065, 0.095)
        valor_atual = valor_zero * ((1.0 - taxa_depr) ** anos_uso)
        
        # Piso de mercado realista por categoria
        if marca in marcas_super_luxo:
            piso = 120000.0
        elif marca in marcas_premium:
            piso = 55000.0
        elif marca in marcas_medias:
            piso = 22000.0
        else:
            piso = 14000.0
            
        valor_atual = max(valor_atual, piso)
        
        marcas_escolhidas.append(marca)
        nomes_escolhidos.append(nome_carro)
        valores_carro.append(valor_atual)
        
    marcas_escolhidas = np.array(marcas_escolhidas)
    nomes_escolhidos = np.array(nomes_escolhidos)
    valores_carro = np.array(valores_carro)
    
    # 5. Cálculo Atuarial do Seguro Anual
    # a) Fator de Idade do condutor
    fatores_idade = np.where(idades < 25, 1.62 + (25 - idades) * 0.02,
                    np.where(idades < 30, 1.25,
                    np.where(idades <= 58, 0.95,
                    1.15)))
    
    # b) Fator de Marca (custo de reposição de peças, índice de sinistro / roubo)
    fatores_marca = []
    for m in marcas_escolhidas:
        if m in marcas_super_luxo:
            fatores_marca.append(1.35)
        elif m in marcas_premium:
            fatores_marca.append(1.25)
        elif m in ['BYD', 'GWM']:
            fatores_marca.append(1.10) # Carros elétricos/híbridos importados (peças/bateria)
        elif m in ['Hyundai', 'Chevrolet', 'Fiat', 'Volkswagen', 'Renault']:
            fatores_marca.append(1.08) # Maior índice de furtos/sinistros urbanos
        elif m in ['Toyota', 'Honda']:
            fatores_marca.append(0.96) # Alta confiabilidade e baixo índice relativo de quebra
        else:
            fatores_marca.append(1.00)
    fatores_marca = np.array(fatores_marca)
    
    # c) Fator ano do carro
    idade_carro = 2025 - anos
    fatores_ano_carro = 1.0 + (idade_carro * 0.012)
    
    # d) Taxa base percentual (~4.8% do valor da tabela)
    taxa_base = np.random.normal(0.048, 0.005, size=n_linhas)
    taxa_base = np.clip(taxa_base, 0.035, 0.065)
    
    # e) Variação estatística individual (bônus, CEP, uso diário, garagem)
    ruido_aleatorio = np.random.normal(1.0, 0.07, size=n_linhas)
    
    valores_seguro = valores_carro * taxa_base * fatores_idade * fatores_marca * fatores_ano_carro * ruido_aleatorio
    valores_seguro = np.maximum(valores_seguro, 1100.0) # Piso de seguro anual
    
    df = pd.DataFrame({
        'Salario': np.round(salarios, 2),
        'Idade': idades,
        'Marca': marcas_escolhidas,
        'Nome': nomes_escolhidos,
        'Ano': anos,
        'Valor_Carro': np.round(valores_carro, 2),
        'Valor_Seguro': np.round(valores_seguro, 2)
    })
    
    return df

if __name__ == '__main__':
    n_linhas = 100000
    df = gerar_dataset(n_linhas)
    caminho_csv = 'seguro_carros.csv'
    df.to_csv(caminho_csv, index=False, sep=',', decimal='.')
    print(f"Dataset gerado com sucesso: {len(df)} linhas salvas em '{caminho_csv}'!")
    print(f"Total de Marcas distintas: {df['Marca'].nunique()}")
    print(f"Total de Modelos (Nome) distintos: {df['Nome'].nunique()}")
    print("\nContagem por Marca (top 15):")
    print(df['Marca'].value_counts().head(15))
    print("\nPrimeiras 10 linhas:")
    print(df.head(10))
