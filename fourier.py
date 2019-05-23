# -*- coding: utf-8 -*-
import numpy as np


def origin_dft(data):
    """
    離散フーリエ変換を行う

    data(numpy):  離散フーリエ変換を行う時間データ
    """
    length = len(data)
    frequency_data_real = np.zeros(length); frequency_data_imag = np.zeros(length)
    for k in range(length):
        b = 0j
        for n in range(length):
            b += data[n] * np.exp(-2j*np.pi*n*k/length)
        frequency_data_real[k] = b.real
        frequency_data_imag[k] = b.imag
    return frequency_data_real, frequency_data_imag



def _calculate_power(n):
    """
    データ数が2の乗数かを確認。2の乗数であれば何乗かを返す。
    
    n(int): データ数
    """
    a = 0
    while n % 2 == 0:
        n = n / 2
        a += 1
    if n == 1.0:
        return a
    else:
        return None

def _fft_algorithm(data):
    """
    バタフライ演算を行う

    data(numpy): バタフライ演算を行うデータ
    """
    primitive = np.exp(-2j * np.pi / len(data))  # 回転因子
    length = int(len(data) / 2)
    data1 = data[:length]; data2 = data[length:]
    new_data = np.zeros(len(data), dtype=np.complex)
    for i in range(length):
        add_data = data1[i] + data2[i]
        new_data[i] = add_data
        sub_data = primitive**i * (data1[i] - data2[i])
        new_data[i+length] = sub_data
    return new_data

def _calculate_reverse_binary(n):
    """
    ビット逆順のソート番号を返す

    n(int): ビット逆順を行うデータ数
    """
    digits = len(format(n-1, 'b'))
    number = np.arange(n)
    binary_list = []

    for i in range(n):
        binary = format(i, 'b').zfill(digits)
        binary = binary[::-1]
        binary_list.append(int(binary))

    binary_sort = np.argsort(binary_list)
    return number[binary_sort]
        

def origin_fft(data):
    """
    高速フーリエ変換を行う。　周波数間引き法を用いている。

    data(numpy): 高速フーリエ変換を行う時間データ
    """

    judgement = _calculate_power(len(data))  # 2の乗数を確認
    data_length = len(data)  # データ数
    data_copy = data.copy()  # データのコピーをとる

    if judgement is None:  # データ数が2の乗数でなければエラーを返す
        raise Exception('Make the number of data a power of 2!')
    
    for i in range(judgement):  # fftの計算
        power = 2 ** i
        split_length = int(data_length / power)
        join_data = np.zeros(len(data), dtype=np.complex)
        for j in range(power):
            start_number = split_length * j; end_number = split_length * (j + 1)
            split_data = data_copy[start_number : end_number]
            new_data = _fft_algorithm(split_data)  # バタフライ演算を行う
            join_data[start_number : end_number] = new_data
        data_copy = join_data.copy()

    butterfly_fft_data = data_copy
    sort_number = _calculate_reverse_binary(data_length)  # ビット逆順の計算を行う

    return butterfly_fft_data[sort_number]  # バタフライ演算のビット逆順による並び替えを行い、周波数データを返す
