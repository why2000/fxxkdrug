import torch
import torchvision
from math import *
import xlrd
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as Tfun

stddeve = 0.01
stddevg = 0.1
stddevw = 0.1
num_years = 6
num_cities = 88
learning_rate = 1e-3
num_step = 3
bias = []

for i in range(num_years):
    b = torch.randn(num_cities)
    bias.append(list(torch.nn.init.normal_(b, mean=0.0, std=stddeve)))
bias = torch.transpose(torch.Tensor(bias),0,1).cuda()

def loadcoor(filename):
    fileinfo = {}
    path = os.path.join('../坐标/',filename)
    with open(path, 'r', encoding='utf-8') as onefile:
        for i, row in enumerate(onefile.readlines()):
            if i == 0:
                continue
            rowsplit = row.split('\t')
            infoneeded = [float(rowsplit[-1].strip('\n').strip('\t').strip(' ')), float(rowsplit[-2].strip('\t').strip(' '))]
            fileinfo[rowsplit[1]] = infoneeded
    return fileinfo

def caculatela(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r * 1000

def loadpol(filename, namedict):
    fileinfo = {}
    path = os.path.join('../人口/',filename)
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
        weight = torch.ones(num_cities, num_cities)
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(weight, mean=2*stddevg, std=stddevg))
        print(weight)
        weight2 = torch.randn((1))
        self.weight2 = torch.nn.Parameter(torch.nn.init.normal_(weight2, mean=2*stddevw, std=stddevw))
        
    def forward(self,x):
        out = torch.mm(self.weight, x) + bias
        return out

class loss(torch.nn.Module):
    def __init__(self):
        self.loss = 0
        super(loss, self).__init__()
    def forward(self,G,out,U,V,D,w):
        loss = 0
        lgg = 0
        for i in range(num_years-1):
            loss += torch.norm((U[:, i]-out[:,i]),p=2)**2
        for i in range(num_cities):
            lgg += log(torch.norm(G[:,i], p=1))
        loss += stddeve**2 * lgg
        loss += torch.norm(G-w*D, p=2)**2*stddevg**2/stddeve**2
        loss += torch.norm(w**2,p=1)*(stddeve**2/stddevw**2)
        loss += torch.sum(G.lt(0).float()*G*100)
        print(loss)
        self.loss = loss
        return loss

net = BasicModule()

net = net.cuda()
cudnn.benchmark = True

optimizer = torch.optim.Adam(net.parameters(),lr=0.5)

def read_data():
    fileloaded = xlrd.open_workbook('../MCM_NFLIS_Data.xlsx')
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

    polfile = os.listdir('../人口/')
    poldict = {}
    for onefile in polfile:
        if onefile.startswith('.'):
            continue
        poldict = dict(poldict, **loadpol(onefile, namedict))
    gids = []
    gids = list(namedict['OH'].values())
    #for key in namedict.keys():
    #    gids.extend(list(namedict[key].values()))
    infected = []
    inf_rate = []
    with open('../gid_order', 'w') as order_table:
        for gid in gids:
            order_table.write(str(gid)+'\n')
    for year in yeardict.keys():
        # print(year)
        if year == 2010 or year == 2017:
            continue
        buf_inf_rate = []
        buf_infected = []
        for gid in gids:
            buf_infected.append(yeardict[year-1][gid])
            buf_inf_rate.append((yeardict[year][gid]-yeardict[year-1][gid])/(poldict[gid]-yeardict[year-1][gid]))
        inf_rate.append(buf_inf_rate)
        infected.append(buf_infected)
    pollist = []
    for gid in gids:
        pollist.append(poldict[gid])

    allfile = os.listdir('../坐标/')
    infolist = {}
    for onefile in allfile:
        if onefile.startswith('.'):
            continue
        infolist[onefile] = loadcoor(onefile)
        
    xs = []
    ys = []

    infoall = {}
    for key in infolist.keys():
        infoall = dict(infoall, **infolist[key])
    for gid in gids:
        county = infoall[gid]
        xs.append(county[0])
        ys.append(county[1])

    rtable = {}
    for x1, y1, gid1 in zip(xs,ys,gids):
        rtable[gid1] = {}
        for x2, y2, gid2 in zip(xs,ys,gids):
            radius = caculatela(x1, y1, x2, y2)
            rtable[gid1][gid2] = radius
    
    dislist = []
    for gid1 in gids:
        bufdis = []
        for gid2 in gids:
            if gid2 == gid1:
                bufdis.append(0)
            else:
                bufdis.append(poldict[gid1]*poldict[gid2]/rtable[gid1][gid2]**2)
        dislist.append(bufdis)
    
    return inf_rate, infected, dislist



def train():
    I = []
    net.train()
    overiter = 0
    for i in range(1000):
        U_raw, V_raw, D_raw = read_data()
        V = torch.autograd.Variable(torch.transpose(torch.Tensor(V_raw),0,1).cuda())
        U = torch.autograd.Variable(torch.transpose(torch.Tensor(U_raw),0,1).cuda())
        D = torch.autograd.Variable(torch.Tensor(D_raw).cuda())
        print("G:")
        print(net.weight)
        print('U:')
        print(list(torch.Tensor.cpu(U[:,-1]).detach().numpy()))
        print(list(torch.Tensor.cpu(U[:,-4]).detach().numpy()))
        print('V:')
        print(list(torch.Tensor.cpu(torch.mv(net.weight, V[:, -1])).detach().numpy()))
        print(list(torch.Tensor.cpu(torch.mv(net.weight, V[:, -4])).detach().numpy()))
        optimizer.zero_grad()
        out = net.forward(V)
        print('out:')
        print(out.shape)
        print('w:')
        print(net.weight2)
        net2 = loss().cuda()
        l = net2(net.weight, out, U, V, D, net.weight2)
        l.backward()
        optimizer.step()
        # if net2.loss < 0.000001:
        #     overiter += 1
            
        # if overiter > 6:
        #     break
    for i in range(2011, 2016):
        It = list(torch.Tensor.cpu(torch.mv(net.weight, V[:, i-2011]) + V[:, i-2011]).detach().numpy())
        I.append(It)
        with open('./result'+str(i+1), 'w') as resfile:
            with open('../gid_order', 'r') as ordfile:
                for res, gid in zip(It, list(ordfile.readlines())):
                    resfile.write(gid.strip('\n')+','+str(res)+'\n')
    prid = list(torch.Tensor.cpu(torch.mv(net.weight, V[:, -1])).detach().numpy())
    print(prid)
    # to_csv(prid, './prid.csv')


train()


