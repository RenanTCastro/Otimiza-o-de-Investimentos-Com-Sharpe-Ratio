import numpy as np
import yfinance as yf

# Função para calcular Sharpe Ratio e último preço
def calculate_sharpe_and_last_price(tickers, risk_free_rate=0, lookback_period='1y'):
    sharpe_ratios = []
    last_prices = []

    for ticker in tickers:
        data = yf.download(ticker, period=lookback_period)
        if data.empty:
            sharpe_ratios.append(np.nan)
            last_prices.append(np.nan)
            continue

        # Calcular retornos diários
        returns = data['Adj Close'].pct_change().dropna()

        # Calcular Sharpe Ratio anualizado
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        sharpe_ratios.append(sharpe)

        # Armazenar o último preço ajustado
        last_prices.append(data['Adj Close'].iloc[-1])

    return np.array(sharpe_ratios), np.array(last_prices)

# Função para gerar a população inicial
def generate_initial_population(tickers, population_size, max_quantity=10):
    n_assets = len(tickers)
    population = np.random.randint(0, max_quantity + 1, size=(population_size, n_assets))
    return population

# Função para calcular o fitness com penalização para carteiras não diversificadas
def calculate_fitness(population, sharpe_ratios, prices, budget, diversification_weight=0.1, max_asset_fraction=0.3):
    fitness = []
    for individual in population:
        total_cost = np.dot(individual, prices.flatten())
        if total_cost > budget:
            fitness.append(-np.inf)
        else:
            # Verificar a restrição de 30% do total investido por ação
            asset_fractions = (individual * prices.flatten()) / total_cost
            if np.any(asset_fractions > max_asset_fraction):
                fitness.append(-np.inf)
                continue

            # Calcular o Sharpe Ratio ponderado
            weighted_sharpe = np.sum(individual * sharpe_ratios)

            # Calcular a penalização para falta de diversificação
            num_assets = np.count_nonzero(individual)  # Número de ativos na carteira
            max_assets = len(individual)
            diversification_penalty = (max_assets - num_assets) / max_assets

            # Fitness ajustado com penalização
            fitness.append(weighted_sharpe - diversification_weight * diversification_penalty)

    return np.array(fitness)

# Função de evolução diferencial
def differential_evolution(population, sharpe_ratios, prices, budget, F=0.8, CR=0.9, generations=100):
    n_individuals, n_assets = population.shape

    for generation in range(generations):
        new_population = []
        for i in range(n_individuals):
            indices = [idx for idx in range(n_individuals) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]

            # Gerar vetor mutante e arredondar para inteiros
            mutant = np.clip(a + F * (b - c), 0, max_quantity)
            mutant = np.round(mutant).astype(int)

            # Cruzamento
            trial = []
            for j in range(n_assets):
                if np.random.rand() < CR or j == np.random.randint(n_assets):
                    trial.append(mutant[j])
                else:
                    trial.append(population[i, j])
            trial = np.array(trial).astype(int)  # Garantir que o trial é inteiro

            # Avaliar fitness do trial e substituir se for melhor
            current_fitness = calculate_fitness([population[i]], sharpe_ratios, prices, budget)[0]
            trial_fitness = calculate_fitness([trial], sharpe_ratios, prices, budget)[0]

            if trial_fitness > current_fitness:
                new_population.append(trial)
            else:
                new_population.append(population[i])

        #print(population)
        population = np.array(new_population)

    # Retornar o melhor indivíduo após todas as gerações
    final_fitness = calculate_fitness(population, sharpe_ratios, prices, budget)
    best_index = np.argmax(final_fitness)
    return population[best_index], final_fitness[best_index]

# Execução Principal
if __name__ == "__main__":
    # Parâmetros iniciais
    tickers = [
        "AAPL",  # Apple
        "GOOGL",  # Alphabet (Google)
        "MSFT",  # Microsoft
        "IBM",  # IBM
        "PEP",  # PepsiCo
        "AMZN",  # Amazon
        "META",  # Meta Platforms (Facebook)
        "TSLA",  # Tesla
        "NFLX",  # Netflix
        "NVDA",  # Nvidia
        "ORCL",  # Oracle
        "AMD",  # Advanced Micro Devices
        "BA",  # Boeing
        "DIS",  # Disney
        "JNJ",  # Johnson & Johnson
        "CVX",  # Chevron
    ]
    risk_free_rate = 0
    lookback_period = '1y'
    budget = 5000
    max_quantity = 100
    population_size = 10
    generations = 100
    F = 0.8
    CR = 0.9

    # Calcular Sharpe Ratio e preços das ações
    sharpe_ratios, prices = calculate_sharpe_and_last_price(tickers, risk_free_rate, lookback_period)

    # Verificar se os dados foram carregados corretamente
    if np.any(np.isnan(sharpe_ratios)) or np.any(np.isnan(prices)):
        print("Erro ao carregar os dados das ações. Verifique os tickers fornecidos.")
    else:
        # Gerar população inicial
        population = generate_initial_population(tickers, population_size, 2)

        # Executar evolução diferencial
        best_portfolio, best_fitness = differential_evolution(
            population, sharpe_ratios, prices, budget, F, CR, generations
        )

        # Exibir resultados
        print("Melhor portfólio encontrado:", best_portfolio)
        print("Fitness do melhor portfólio (Sharpe Ratio):", best_fitness)
        print("Custo total do portfólio:", np.dot(best_portfolio, prices.flatten()))
        print(prices)