"""
Project Iterative Game Theory

Developer: Stanislav Alexandrovich Ermokhin

"""

import pprint
import pandas as pd
import pulp as pl


def prudent(lst):
    """

    :param lst: list
    :return: int
    """

    mins = dict()

    for j in range(len(lst)):
        mins[j] = min(lst[j])

    for j in mins:
        if mins[j] == max([mins[key] for key in mins]):
            return j


def play(k, strategies):
    """

    :param k: int
    :param strategies: list
    :return: dict
    """

    results = dict()

    for num in range(k):
        if num == 0:
            i = prudent(strategies)
            uaj_wave = strategies[i]
            alpha_wave = min(uaj_wave) / (num + 1)
            j = uaj_wave.index(min(uaj_wave))
            uai_wave = [item[j] for item in strategies]
            beta_wave = max(uai_wave) / (num + 1)
            nu_wave = (alpha_wave + beta_wave) / 2
            results[str(num + 1)] = {'k': str(num + 1),
                                     'i': i + 1,
                                     'ua(j)_wave': uaj_wave,
                                     'alpha_wave': alpha_wave,
                                     'j': j + 1,
                                     'ua(i)_wave': uai_wave,
                                     'beta_wave': beta_wave,
                                     'nu_wave': nu_wave,
                                     'delta': '',
                                     'epsilon, %': ''}

        else:
            previous_uai_wave = results[str(num)]['ua(i)_wave']
            i = previous_uai_wave.index(max(previous_uai_wave))
            previous_uaj_wave = results[str(num)]['ua(j)_wave']
            uaj_wave = [previous_uaj_wave[q] + strategies[i][w]
                        for q in range(len(previous_uaj_wave))
                        for w in range(len(strategies[i]))
                        if q == w]
            alpha_wave = min(uaj_wave) / (num + 1)
            j = uaj_wave.index(min(uaj_wave))
            vertical = [item[j] for item in strategies]
            uai_wave = [previous_uai_wave[q] + vertical[w]
                        for q in range(len(previous_uai_wave))
                        for w in range(len(vertical))
                        if q == w]
            beta_wave = max(uai_wave) / (num + 1)
            nu_wave = (alpha_wave + beta_wave) / 2

            delta = abs(abs(nu_wave) - abs(results[str(num)]['nu_wave']))
            epsilon = delta / abs(nu_wave) * 100

            results[str(num + 1)] = {'k': str(num + 1),
                                     'i': i + 1,
                                     'ua(j)_wave': uaj_wave,
                                     'alpha_wave': alpha_wave,
                                     'j': j + 1,
                                     'ua(i)_wave': uai_wave,
                                     'beta_wave': beta_wave,
                                     'nu_wave': round(nu_wave, 2),
                                     'delta': round(delta, 2),
                                     'epsilon, %': round(epsilon, 2)}

    return results


def optimization_play(strategies):
    """

    :param strategies: list
    :return: dict
    """

    greater_zero_status = True

    for item in strategies:
        for value in item:
            if value <= 0:
                greater_zero_status = False

    if not greater_zero_status:
        min_min = min([min(item)
                       for item in strategies])
        plus = abs(min_min - 1)

        for item in strategies:
            for j in range(len(item)):
                item[j] += plus

    n = len(strategies)
    m = len(strategies[0])
    variables = [pl.LpVariable("p" + str(k), lowBound=0)
                 for k in range(m)]
    solver = pl.PULP_CBC_CMD()
    model = pl.LpProblem("Game", pl.LpMinimize)
    model += sum(variables)
    for _i in range(m):
        model += sum([strategies[_i][k] * variables[k]
                      for k in range(n)]) >= 1

    result1 = model.solve(solver=solver)
    model_q = pl.LpProblem("B Game", pl.LpMaximize)

    variables_q = [pl.LpVariable("q" + str(k), lowBound=0)
                   for k in range(n)]
    model_q += sum(variables_q)
    for _j in range(n):
        model_q += sum([strategies[_j][k] * variables_q[k]
                        for k in range(m)]) <= 1
    result2 = model_q.solve(solver=solver)

    if result1 and result2:
        mu = 1 / model.objective.value()
        x_y_mu = {'x': [mu * p.value()
                        for p in variables],
                  'y': [mu * q.value()
                        for q in variables_q],
                  'mu': mu if greater_zero_status else mu - plus}

        return x_y_mu

    else:
        return dict()


def write_smth(obj, name, action, type_play_solution):
    """

    :param obj: dict
    :param name: str
    :param action: str
    :param type_play_solution: int
    :return: bool or str
    """

    try:
        df = pd.DataFrame.from_dict(obj)
        if type_play_solution == 1:
            df.set_index('k', inplace=True)
            df.sort_values(by=['k'])

        if action == '2':
            with open(name + '.csv', 'w') as a:
                a.write(df.to_csv())

        elif action == '3':
            with open(name + '.html', 'w') as a:
                a.write(df.to_html())

        elif action == '4':
            df.to_excel(excel_writer=name + '.xlsx')

        return True

    except BaseException as error:
        return error


strats = list()
status = True

while status:

    try:
        status = False

        size_1 = int(input('Number of strategies for Player 1: '))
        type_play = int(input('1: Iterative\n'
                              '2: Linear optimization\n'
                              'Enter: '))

        if size_1 < 2:
            raise ValueError
        for i in range(size_1):
            strats.append(list(map(int, input('Utilities for Strategy ' + str(i + 1) + ' (use spaces): ').split())))
        check_size = [len(item)
                      for item in strats]
        if max(check_size) != min(check_size) \
                or (len(check_size) == 1 and not check_size[0]):
            raise ValueError

        if type_play == 1:
            iterations = int(input('Number of iterations: '))
            results = play(iterations, strats)
            added_results = {'x': {i + 1: 0 for i in range(size_1)},
                             'y': {j + 1: 0 for j in range(len(strats[0]))}}

            for key in results:
                added_results['x'][results[key]['i']] += 1
                added_results['y'][results[key]['j']] += 1

            added_results.update({'x': [added_results['x'][key] / iterations for key in added_results['x']],
                                  'y': [added_results['y'][key] / iterations for key in added_results['y']]})

        elif type_play == 2:
            results = optimization_play(strats)

        action = input('0: EXIT\n'
                       '1: show results\n'
                       '2: write results to csv\n'
                       '3: write to html\n'
                       '4: write to excel\nEnter: ')

        if action == '0':
            print('Exiting...')

        elif action == '1' and type_play == 1:
            pprint.pprint((results[str(iterations)], added_results))

        elif action == '1' and type_play == 2:
            pprint.pprint(results)

        elif action in {'2', '3', '4'}:
            filename = input('Filename (no extension): ')
            print(write_smth(results, filename, action, type_play))

        else:
            print('Exiting...')

    except ValueError:
        print('Error')
        status = True
