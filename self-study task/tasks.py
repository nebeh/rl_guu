#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
n = 1 # номер варианта
np.random.seed(n)

vendors = np.random.randint(50,70)
сonsumer = np.random.randint(50,70)
V = np.random.randint(1,300,vendors)
C = np.random.randint(1,300,сonsumer)

cost = np.random.rand(vendors,сonsumer)*6