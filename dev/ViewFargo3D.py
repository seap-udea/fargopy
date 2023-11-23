#Imports --------------------------------------------------------------#

import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.patches  import Ellipse

import warnings
warnings.filterwarnings("ignore")


#output_folder ="/home/mmontesinos/Simulations/2020/Fargo3D/fargo3D-RT/outputs/planetesimalsRT_fb0/"
#output_folder = "/mnt/data1/mmontesinos/2021/fargo3d/model_f1/"  # OK

#output_folder = "/mnt/data1/mmontesinos/2021/fargo3d/model_f0_energy/"  # NO!
#output_folder = "/mnt/data1/mmontesinos/2021/fargo3d/model_f1_energy/"  # NO!

#output_folder = "/mnt/data1/mmontesinos/2022/f4/"                 # OK
#output_folder = "/mnt/data1/mmontesinos/2022/f2/"                 # OK
#output_folder = "/mnt/data1/mmontesinos/2022/f1/"                 # OK
#output_folder = "/mnt/data1/mmontesinos/2022/f3/"                 # NO!

#output_folder ="/mnt/data1/mmontesinos/2021/fargo3d/paper_f1/" # simulacion mala!
#output_folder = "/mnt/data1/mmontesinos/2021/fargo3d/paper_f0/" NO!
#output_folder = "/mnt/data1/mmontesinos/paperFargo3D/M1_f0/" # NO!

output_folder = "/home/jzuluaga/public/outputs/p3diso_f0/"

"""
snapshot = int(input("Enter output: "))
print(output_folder, "output:", snapshot)
"""

snapshot = 2

##Retorna siempre str, recordad tranformar si necesita un int o float
#variables_par = np.genfromtxt(output_folder+"/variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()#Lee archivo variable.pary convierte a string       

#Retorna siempre str, recordad tranformar si necesita un int o float
variables_par = np.genfromtxt(output_folder+"/variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()#Lee archivo variable.pary convierte a string       


#Pametros--------------------------------------------------------------#
def P(parametro):
        parametros, valores = [],[]                                                                                                                                                                                                                             #Reparte entre parametros y valores
        for posicion in variables_par:                                                                                                                                                                                                                  #
                parametros.append(posicion[0].decode("utf-8"))                                                                                                                                                                          #
                valores.append(posicion[1].decode("utf-8"))                                                                                                                                                                                     #
        return valores[parametros.index(parametro)]     


#Inputs ---------------------------------------------------------------#
#Formato : [PLANO,MAPA,CORTE,SNAPSHOT]
#Planos  : XY , YZ
#Mapa    : 1 = Densidad , 2 = Temperatura, 3 = Vphi , 4 = Vr, 5 = Vz
#Corte   : (Depende del plano): Para XY es un corte en Z (theta) donde el mid-plane estaria en NZ/2. Para YZ es un corte en X (phi) donde el planeta se ubica en NX/2
#Snapshot: Snapshot
#             [[Plano, Mapa, Corte, Snap]]


mid_z =  int(int(P("NZ"))/2)
mid_x =  int(int(P("NX"))/2)

snapshot1 = snapshot
snapshot2 = snapshot
snapshot3 = snapshot

show_planes = [["XY",1,mid_z,snapshot1],["YZ",1,mid_x,snapshot2],["YZ",1,mid_x,snapshot3]]

log    = True                           #Aplica para todos los plot por igual
limits = False                          #Aplica para todos los plot por igual
limit_min,limit_max = 2,4       #No se puede mezclar mapas para los limites.

save = False

def CFG():
        global Rp,Mp
        #!! IMPORTANTE !! : Los valores en archivo .cfg deben estar separados por TAB y no espacidos
        archivo = np.loadtxt(output_folder+'../../'+P("PLANETCONFIG"),dtype="str",delimiter='   ')
        valores_planet = []
        for valor in archivo:
                if valor != "":
                        valores_planet.append(valor)
        Rp,Mp = float(valores_planet[1]),float(valores_planet[2])       
#CFG()

#Planet
print("Leyendo Planeta ...")
planet  = np.genfromtxt(output_folder+"planet0.dat")				
xp      = planet[snapshot][1] ; yp      = planet[snapshot][2]
zp      = planet[snapshot][3] ; Mp      = planet[snapshot][7]
Rp      = np.sqrt((xp**2)+(yp**2)+(zp**2))

print(Rp)

#Rp = 1.0
#Mp = 1e-3

#Unidades--------------------------------------------------------------#
def units():
        global unit_mass,unit_length,gamma,mu,mp,kb,G,Rgas
        global unit_surf_density,unit_volm_density,unit_time,unit_disktime,unit_orbits
        global unit_temperature,unit_energy,unit_velocity
        #Constantes -------------------------------------------------------#
        kb = 1.380650424e-16                                                                                            #erg/K    | erg = gr cm2 / s2
        mp = 1.672623099e-24                                                                                            #gr
        mu = 2.35                                                                                                                       #mol
        gamma = float(P("GAMMA"))                                                                                       #
        G = 6.67259e-8                                                                                                          #cm3/gr/s2
        Rgas = 8.314472e7                                                                                                               #erg /K/mol
        unit_mass   = 1.9891e33                                                                                         #gr
        unit_length = 1.49598e13                                                                                        #cm
        
        #Unidades ---------------------------------------------------------#
        unit_surf_density = (unit_mass/unit_length**2)                                                  #gr/cm2
        unit_volm_density = (unit_mass/unit_length**3)                                                  #gr/cm3
        unit_temperature  = ((G*mp*mu)/(kb))*(unit_mass/(unit_length))                  #K
        unit_time     = np.sqrt( pow(unit_length,3.) / G / unit_mass) / 3.154e7 #yr
        unit_disktime = float((float(P("NINTERM"))*float(P("DT"))))*unit_time   #yr/snapshot    [Las unidades de tiempo del disco se toman en R=1]
        unit_energy = unit_mass*(unit_length**2)/(unit_time*3.154e7)**2/ unit_length**3.                        #erg
        unit_period   = unit_time*2*np.pi*np.sqrt(Rp**3)                                                #yr
        unit_orbits   = unit_disktime/unit_period                                                               #orbita/snapshot        [Fraccion de la orbita del planeta]
        unit_velocity = 1e-5*unit_length/float(unit_time*3.154e7)                               #km/s
units()

#Dominios -------------------------------------------------------------#
domain_x = np.genfromtxt(output_folder+"/domain_x.dat")
domain_y = np.genfromtxt(output_folder+"/domain_y.dat")[3:-3] #[3:-3] porque dominio Y utiliza celdas "fantasmas"
domain_z = np.genfromtxt(output_folder+"/domain_z.dat")[3:-3] #[3:-3] porque dominio Z utiliza celdas "fantasmas"

#Grillas --------------------------------------------------------------#
def Grilla_XY():
        P,R  = np.meshgrid(domain_x,domain_y) #Entrega radio y co-latitud
        X,Y  = R*np.cos(P), R*np.sin(P)
        return X,Y
        
def Grilla_YZ():
        #Coordendas esfericas inversas : R = r sin(theta), Phi = Phi, z = r cos(theta) 
        #En este caso theta = z angulo de colatitud, mientras phi = x angulo azimutal
        
        R,T = np.meshgrid(domain_y, domain_z) #Entrega radio y azimut
        Y,Z = R*np.sin(T),R*np.cos(T)
        return Y,Z
        
#Definicion temperatura -----------------------------------------------#
def get_temperature(type_map,cut_z,snapshot,unidad):
        density = np.fromfile(output_folder+"gasdens"+str(snapshot)+".dat").reshape(int(P("NZ")),int(P("NY")),int(P("NX"))) #Entrega: NZ matrices de NY por NX.
        energy  = np.fromfile(output_folder+"gasenergy"+str(snapshot)+".dat").reshape(int(P("NZ")),int(P("NY")),int(P("NX"))) #Entrega: NZ matrices de NY por NX.
        vtheta = np.fromfile(output_folder+"gasvx"+str(snapshot)+".dat").reshape(int(P("NZ")),int(P("NY")),int(P("NX")))
        
        #FALTA DEFINIR BIEN LA UNIDAD DE TEMPERATURA !! CREO ...
        press = (gamma - 1.0)*energy*unit_energy
        temperature = mu*press /(density*unit_volm_density *Rgas)
        cs = np.sqrt(gamma*press/(density *unit_volm_density ))
        #print(temperature)
        H = cs/vtheta
        print(temperature.max())
        a=2
        a=9
        

        return temperature
        
#Funciones Plot -------------------------------------------------------#
def plot_XY(type_map,cut_z,snapshot,unidad):
        global cmap                                                                                                                     #globlaliza la variable cmap
        if type_map == 2:                                                                                                       #Condicion para obtener Temperatura, (depende de dos mapas densidad y energia)
                data = get_temperature(type_map,cut_z,snapshot,unidad)
        else:                                                                                                                           #cualquier otro mapa que no sea temperatura
                data = np.fromfile(output_folder+types_map[type_map]+str(snapshot)+".dat").reshape(int(P("NZ")),int(P("NY")),int(P("NX"))) #Entrega: NZ matrices de NY por NX.
                data = data*types_units[unidad]

        if type_map == 3 or type_map == 4 or type_map == 5: cmap = "seismic"#solo para mapas de velocidades
        else: cmap = "viridis"                                                                                          #cualquier otro mapa
        
        X,Y = Grilla_XY()
        data = data[cut_z]                                                                                                      #Selecciona una de las NZ matrices creadas con reshape
        
        if log == True:                                                                                                         #Condicion para mapa en escala logaritmica
                data = np.log10(data)
                
        return X,Y,data

def plot_YZ(type_map,cut_x,snapshot,unidad):
        global cmap                                                                                                                     #globlaliza la variable cmap
        if type_map == 2:                                                                                                       #Condicion para obtener Temperatura, (depende de dos mapas densidad y energia)
                data = get_temperature(type_map,cut_x,snapshot,unidad)          
        else:
                data = np.fromfile(output_folder+types_map[type_map]+str(snapshot)+".dat").reshape(int(P("NZ")),int(P("NY")),int(P("NX"))) #Entrega: NZ matrices de NY por NX.
                data = data*types_units[unidad] 
        
        if type_map == 3 or type_map == 4 or type_map == 5: cmap = "seismic"#solo mapas de velocidades
        else: cmap = "viridis"                                                                                          #cualquier otro mapa
                
        Y,Z = Grilla_YZ()
        data = data[:,:,cut_x]                                                                                          #Toma las NZ matrices con NY valores de algun cut_x
        
        if log == True:                                                                                                         #Condicion para mapa en escala logaritmica
                data = np.log10(data)
                
        return Y,Z,data

#Colobar --------------------------------------------------------------#
def cbar_ax(im,i,title_cbar):
        p    = ax[i].get_position().get_points().flatten()                      #entrega (x0,y0,x1,y1) de la figura selecionada
        
        cax  = fig.add_axes([p[2]+0.005, p[1], 0.005, p[3]-p[1]])       #recibe (x0,y0,ancho,largo) de la barra
        cbar = fig.colorbar(im,ax=ax[i],cax=cax)                                        #cax distribuye donde y como ira la barra de colores
        
        cbar.set_label(title_cbar,fontsize=12)
        cbar.ax.tick_params(axis="y", direction="in")
        
#Listas de datos ------------------------------------------------------# (Editable, se pueden agregar mapas, energia, velocidades ... etc)
types_map    = ["","gasdens"        ,"gasTemperature"   ,"gasvx"       ,"gasvy"       ,"gasvz"      ]
types_units  = ["",unit_surf_density,"unit_temperature" ,unit_velocity ,unit_velocity ,unit_velocity]

type_bar     = ["",r"Gas density [gr $cm^{-2}$]"             ,r"Gas Temperature [K]"              ,r"$V_{\phi}$ [km/s]" ,r"$V_{r}$ [km/s]" ,r"$V_{z}$ [km/s]"]
type_bar_log = ["",r"Gas density $(log_{10})$ [gr $cm^{-2}$]",r"Gas Temperature $(log_{10})$ [K]" ,"No se puede !!"     ,"No se puede !!"  ,"No se puede !!" ]

#----------------------------------------------------------------------#
#Codigo ---------------------------------------------------------------#
#Fig distribucion de filas y columnas ---------------------------------#
n_column = len(show_planes)
n_row = 1

if n_column == 4:
        n_column = 2
        n_row = 2

elif n_column >4:
        n_column = 3
        n_row = len(show_planes)//3
        if len(show_planes)%3 !=0:
                n_row = len(show_planes)//3+1

#Opcines de Tamanos ---------------------------------------------------#
bottom,left,top,right = 0.09,0.07,1.- 0.09,1. - 0.1
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )

wspace = 0.45
hspace = 0.3
altura_figura  = 4.8*n_row
figheight = altura_figura
aspect = 1.
figwidth  = (n_column + (n_column-1)*wspace)/float((n_row+(n_row-1)*hspace)*aspect)*figheight*fisasp

if n_column == 1:
        altura_figura  = 9.4
        aspect = 1.
        bottom,left,top,right = 0.07,0.1,1.- 0.07,1. - 0.15
        fisasp = (1-bottom-(1-top))/float( 1-left-(1-right))
        figheight = altura_figura
        figwidth  = (n_column + (n_column-1)*wspace)/float((n_row+(n_row-1)*hspace)*aspect)*figheight*fisasp
        
fig, axes = plt.subplots(nrows=n_row, ncols=n_column, figsize=(figwidth, figheight),squeeze=False) #Squeeze = False, para no tener problemas con plots de 1x1
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
ax = axes.flatten()

#Iteracion ------------------------------------------------------------#
k = 0
for plot in show_planes:
        
        #Plano XY ---------------------------------------------------------#
        if   plot[0] == "XY":
                #Plot principal -----------------------------------------------#
                X,Y,data = plot_XY(plot[1],plot[2],plot[3],plot[1])
                if limits == True:
                        im1  = ax[k].pcolormesh(X,Y,data,cmap=cmap,vmin = limit_min , vmax=limit_max)
                else:
                        im1  = ax[k].pcolormesh(X,Y,data,cmap=cmap)
                        
                #Ajuste encuadre ----------------------------------------------# Es importante que el ajuste de encuadre este antes de asignar un colobar. De lo contrario, habra que modificar el "cax" de la colorbar (cambia su tamaño)
                ax[k].axis("image")
                
                #Colorbar -----------------------------------------------------#
                if log == True:
                        cbar_ax(im1,k,type_bar_log[plot[1]])
                else:
                        cbar_ax(im1,k,type_bar[plot[1]])
                        
                #Opciones del plot --------------------------------------------#
                #Labels -------------------------------------------------------#
                snapshot = plot[3]
                ax[k].set_title("Time: "+str(round(snapshot*unit_disktime,2))+" yr "+"  "+"Orbits: "+str(round(snapshot*unit_orbits,2)) +"  "
                                                +r"   $\theta$ = "+str(round(domain_z[plot[2]]/np.pi,2))+str(r"$\pi$ rad"),fontsize=12)
                ax[k].set_xlabel("x [au]",fontsize=12)
                ax[k].set_ylabel("y [au]",fontsize=12)
                
                #Lineas externas (solo estetica) ------------------------------#
                ax[k].add_artist(Ellipse(xy=(0,0), width = float(P("YMIN"))*2, height = float(P("YMIN"))*2,fill=False,lw=0.8,color="black"))            #Disco interior
                ax[k].add_artist(Ellipse(xy=(0,0), width = float(P("YMAX"))*2, height = float(P("YMAX"))*2,fill=False,lw=0.8,color="black"))            #Disco exterior


        #Plano YZ ---------------------------------------------------------#
        elif plot[0] == "YZ":
                #Plot principal -----------------------------------------------#
                Y,Z,data = plot_YZ(plot[1],plot[2],plot[3],plot[1])
                if limits == True:
                        im2  = ax[k].pcolormesh(Y,Z,data,cmap=cmap,vmin = limit_min, vmax = limit_max)
                else:
                        im2  = ax[k].pcolormesh(Y,Z,data,cmap=cmap)
                        
                #Ajuste encuadre ----------------------------------------------# Es importante que el ajuste de encuadre este antes de asignar un colobar. De lo contrario, habra que modificar el "cax" de la colorbar (cambia su tamaño)
                ax[k].axis("on") #On: para que encuadre bien junto con los planos XY
                dy,dz = 0.02,0.17
                ax[k].axis([np.min(Y)-dy,np.max(Y)+dy,np.min(Z)-dz,np.max(Z)+dz])
                
                #Colorbar -----------------------------------------------------#
                if log == True:
                        cbar_ax(im2,k,type_bar_log[plot[1]])
                else:
                        cbar_ax(im2,k,type_bar[plot[1]])
                
                #Altura fisica del disco --------------------------------------#
                domain_H = float(P("ASPECTRATIO"))*((domain_y[::2])**(1.0 + float(P("FLARINGINDEX"))))
                ax[k].plot(domain_y[::2],domain_H,color="white",lw=0.3)
                ax[k].plot(domain_y[::2],-domain_H,color="white",lw=0.3)
                
                #Opciones del plot --------------------------------------------#
                #Labels -------------------------------------------------------#
                snapshot = plot[3]
                ax[k].set_title("Time: "+str(round(snapshot*unit_disktime,2))+" yr "+"  "+"Orbits: "+str(round(snapshot*unit_orbits,2)) +"  "
                                                +r"$\phi$ = "+str(round(domain_x[plot[2]],2))+str(" rad"),fontsize=12)
                ax[k].set_xlabel("r [au]",fontsize=12)
                ax[k].set_ylabel("z [au]",fontsize=12)
                
                #Lineas externas (solo estetica) ------------------------------#
                ax[k].add_artist(Ellipse(xy=(0,0), width = float(P("YMIN"))*2, height = float(P("YMIN"))*2,fill=False,lw=0.8,color="black"))    #Disco interior
                ax[k].add_artist(Ellipse(xy=(0,0), width = float(P("YMAX"))*2, height = float(P("YMAX"))*2,fill=False,lw=0.8,color="black"))    #Disco exterior
                ax[k].plot(Y[0],Z[0],lw=1,color="black")
                ax[k].plot(Y[0],Z[-1],lw=1,color="black")
                
        
        #Opciones generales -----------------------------------------------#
        #Ticks ------------------------------------------------------------#
        color_line="black"
        ax[k].set_facecolor("white")
        ax[k].xaxis.set_minor_locator(tck.AutoMinorLocator(2))          #Agrega 2 ticks pequenos entre los ticks eje x
        ax[k].yaxis.set_minor_locator(tck.AutoMinorLocator(2))          #Agrega 2 ticks pequenos entre los ticks eje y
        ax[k].tick_params(axis="both",which="minor",direction="in",length=2,width=1,color=color_line) #Ajuste de ticks menores
        ax[k].tick_params(axis="both",which="major",direction="in",length=3,width=1,color=color_line) #Ajuste de ticks mayores

        for spine in ax[k].spines.values(): #Color del encuadre
                spine.set_edgecolor(color_line) 
        #------------------------------------------------------------------#
        
        k+=1

if save ==True: plt.savefig("/home/jmgarrido/Papers/3DPlots/"+"5"+".png",dpi=200)

plt.show(),plt.close()


