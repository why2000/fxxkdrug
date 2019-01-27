import torch
import torchvision
from math import *
import xlrd
import os

stddev = 1.0
num_years = 5
num_cities = 88
learning_rate = 1e-3
num_step = 3
bias = []
for i in range(num_years):
    b = torch.randn(num_cities)
    bias.append(list(torch.nn.init.normal_(b, mean=0.0, std=stddev)))
bias = torch.transpose(torch.Tensor(bias),0,1)

def loadpol(filename, namedict):
    fileinfo = {}
    path = os.path.join('./2018_MCMProblemC_DATA/人口/',filename)
    with open(path, 'r', encoding='utf-8') as onefile:
        for i, row in enumerate(onefile.readlines()):
            rowsplit = row.split(',')
            infoneeded = int(rowsplit[-1].strip('\n'))
            countyname = rowsplit[0].upper()
#             print(countyname)
            if countyname.endswith(' COUNTY'):
                countyname = countyname[:-7]
#                 print(countyname)
            if countyname in ['RICHMOND']:
                continue
            if countyname.endswith(' CITY'):
                countyname = countyname[:-5]
#                 print(countyname)
            if countyname.startswith('CITY OF '):
                if not countyname[8:] in ['FAIRFAX','FRANKLIN']:
                    countyname = countyname[8:]
#                     print(countyname)
                else:
                    countyname = countyname[8:] +' CITY'
#                     print(countyname)
#             print(countyname+str(namedict[filename][countyname]))
            if countyname not in namedict[filename].keys():
                # print(countyname)
                continue
            fileinfo[namedict[filename][countyname]] = infoneeded
    return fileinfo

def to_csv(input_list, filepath):
    with open(filepath, 'w') as wfile:
        for row in input_list:
            wfile.write(','.join(map(str,row))+'\n')

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        bias = []
        weight = torch.nn.Parameter(torch.randn(num_cities, num_cities))
        self.weight = torch.nn.init.normal_(weight, mean=0.0) 
        
    def forward(self,x):
        out = torch.mm(self.weight, x) + bias
        return out

class loss(torch.nn.Module):
    def __init__(self):
        super(loss, self).__init__()
    def forward(self,out, U, V):
        loss = 0
        for i in range(num_years-1):
            loss += torch.norm((U[:, i]-torch.mv(net.weight, V[:, i])),p=2,dim=None)**2
        loss = loss/(stddev**2)
        return loss

net = BasicModule()
optimizer = torch.optim.SGD(net.parameters(),lr=0.001)

def read_data():
    fileloaded = xlrd.open_workbook('./2018_MCMProblemC_DATA/MCM_NFLIS_Data.xlsx')
    table = fileloaded.sheets()[1]
    placelist = []
    placedict = {}
    for i in range(1, 24062):
        rowv = table.row_values(rowx = i)
        placename = rowv[5]
        time1 = rowv[7]
        placelist.append(rowv)
        if placename not in placedict.keys():
            placedict[rowv[5]] = time1
        else:
            placedict[rowv[5]] += time1
        yearlist = []
        yeardict = {}
    for i in range(1, 24062):
        rowv = table.row_values(rowx = i)
        year = int(rowv[0])
        time = int(rowv[7])
        place = rowv[5]
        drugname = rowv[6]
        yearlist.append(rowv)
        if drugname != 'Heroin':
            continue
        if year not in yeardict.keys():
            yeardict[year] = {}
            yeardict[year][place] = time
        elif place not in yeardict[year].keys():
            yeardict[year][place] = time
        else:
            yeardict[year][place] += time
    for year in yeardict.keys():
        for place in placedict.keys():
            if place not in yeardict[year].keys():
                yeardict[year][place] = 0
    
    namelist = []
    namedict = {}
    for i in range(1, 24062):
        rowv = table.row_values(rowx = i)
        statename = rowv[1]
        countyname = rowv[2]
        if countyname.endswith(' CITY') and not countyname[:-5] in ['FAIRFAX', 'FRANKLIN', 'RICHMOND']:
            countyname = countyname[:-5]
        sgid = rowv[5]
        namelist.append(rowv)
        if statename not in namedict.keys():
            namedict[statename] = {}
            namedict[statename][countyname] = sgid
        else:
            namedict[statename][countyname] = sgid

    polfile = os.listdir('./2018_MCMProblemC_DATA/人口/')
    poldict = {}
    for onefile in polfile:
        if onefile.startswith('.'):
            continue
        poldict = dict(poldict, **loadpol(onefile, namedict))
    gids = list(namedict['OH'].values())
    infected = []
    inf_rate = []
    to_csv(gids, './gid_order')
    for year in yeardict.keys():
        # print(year)
        if year == 2010 or year == 2017 or year == 2016:
            continue
        buf_inf_rate = []
        buf_infected = []
        for gid in gids:
            buf_infected.append(yeardict[year][gid])
            buf_inf_rate.append((yeardict[year][gid]-yeardict[year-1][gid])/(poldict[gid]-yeardict[year-1][gid]))
        inf_rate.append(buf_inf_rate)
        infected.append(buf_infected)
    return inf_rate, infected

def train():
    net.train()
    for i in range(2):
        U_raw, V_raw = read_data()
        V = torch.autograd.Variable(torch.transpose(torch.Tensor(V_raw),0,1))
        U = torch.autograd.Variable(torch.transpose(torch.Tensor(U_raw),0,1))
        optimizer.zero_grad()
        out = net.forward(V)
        net2 = loss()
        l = net2(out, U, V)
        l.backward()
        optimizer.step()
        
    prid = list(torch.mv(net.weight, V[:, -1]).detach().numpy())
    print(prid)
    to_csv(prid, './prid.csv')


train()

