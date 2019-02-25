# ELIMINATE QUOTE MARKS FROM PRINTING
noquote("a") # For example

# CHANGE NAME OF A VARIABLE
names(df)[names(df) == 'old_var_name'] <- 'new_var_name'

# SAMPLING / SAMPLE RANDOM ROWS / SELECT RANDOM ROWS / RANDOM ROWS SELECTION
df[sample(nrow(df), 3), ] # Select 3 random rows of df

# SAVE AN RDA FILE IN R
save(object1, object2, object3, file = "route/mydata.rda")

# PASTE A VECTOR OF STRINGS OR CHARACTERS
vector_strings <- c("paste", "this", "phrase")
paste(vector_strings, collapse = '')

                            #FORMULARIO

#data <- read.table("cup98lrn.txt", header = TRUE, sep = ",", quote="", fill = T, stringsAsFactors = T, na.strings=c(""," ","  ","   ","NA"))


#CABECERA DE SCRIPTS / HEAD DE SCRIPTS /
############################################################################
# Título: XXX
# Fecha: XX de nombre_mes de XXXX
# Tratamiento: Train ó Tes
# Tipo de datos: XX
############################################################################

#SEPARADOR DE TROZOS DE CÓDIGO
############################################################################
                    # NOMBRE_DE_LA_PARTE
############################################################################

#FUNCIÓN - ELIMINAR VARIABLES FACTOR CON SÓLO 1 NIVEL / 1 LEVEL 
mas_de_un_nivel <- function(data) {
  cat <- sapply(data, is.factor)
  w <- Filter(function(x) nlevels(x)>1, data[,cat])
  otras <- data[!sapply(data,is.factor)]
  data <- cbind(w,otras)
  return(data)
}

#FUNCIÓN - ACTUALIZAR FACTORES
actualizar_factores <- function(data) {
  cat <- sapply(data, is.factor) #Calcula las variables categoricas
  A <- function(x) factor(x)
  data[ ,cat] <- data.frame(apply(data[ ,cat],2, A))
  return(data)
}

#CONFIGURAR PROXY
Sys.setenv(http_proxy="http://amartinezsistac:POI17asd@proxy.indra.es:8080/")

#ACTUALIZACIÓN DE VARIABLES FACTOR
cat <- sapply(data, is.factor) #Calcula las variables categoricas
data[ ,cat] #Te enseña las variables categoricas por pantalla
names(data[ ,cat]) #Enseña cuáles son sus nombres
A <- function(x) factor(x)
data[ ,cat] <- data.frame(apply(data[ ,cat],2, A))
str(data)
ncol(data)
nrow(data)

#ELIMINAR VARIABLES FACTOR CON SÓLO 1 NIVEL / 1 LEVEL
cat <- sapply(data, is.factor)
w <- Filter(function(x) nlevels(x)>1, data[,cat])
otras <- data[!sapply(data,is.factor)]
data <- cbind(w,otras)
ncol(data)
nrow(data)
names(data)
str(data)

#DESCRIPTIVO CON TODAS LAS VARIABLES / STR / MOSTRAR TODAS LAS VARIABLES EN STR
str(data, list.len=ncol(data))

#SELECCIONAR VARIABLES CATEGÓRICAS/FACTOR
cat <- sapply(data, is.factor) #Calcula las variables categoricas
categoricas <- data[ ,cat]

#SELECCIONAR VARIABLES NUMÉRICAS
num <- sapply(data, is.numeric) #Calcula las variables categoricas
numericas <- data[ ,num]

#SELECCIONAR VARIABLES CHARACTER
char <- sapply(data, is.character) #Calcula las variables que no son caracter
character <- data[ ,char]

#SELECCIONAR VARIABLES NO CHARACTER
no_char <- !sapply(data, is.character) #Calcula las variables que no son caracter
no_character <- data[ ,no_char]

# FILTER DATA FRAME BY COLUMN
data[,-c(which(colnames(data)=="VAR1"), which(colnames(data)=="VAR2")) ]
              
#RECODIFICAR VARIABLES CATEGÓRICAS (MAPVALUES)
data$variable_a_recodificar <- mapvalues(data$variable_a_recodificar, c("valor1","valor2", ...), c("valor1_nuevo","valor2_nuevo",...))

#ELIMINACIÓN DE VARIAS COLUMNAS EN UNA SOLA ORDEN - ELIMINACIÓN DE COLUMNAS MASIVAS
temporal <- names(data) %in% c("columna_a_eliminar1","columna_a_eliminar2", ...)
data <- data[!temporal]

#ELIMINACIÓN DE REDUNDANCIA O CORRELACIÓN
nums <- data[sapply(data,is.numeric)]
head(nums)
names(nums)
ncol(nums)
library(corrplot)
matriu_correlacions1 <- cor(nums)
corrplot(matriu_correlacions1, method = "circle",tl.cex = 0.5)

#ELIMINACIÓN DE OUTLIERS
nrow(data)
coef <- round(nrow(data)*(5/1000))
coef
rownames(data) <- seq_len(nrow(data))
(ou<-outlierTest(nombre_del_modelo,n.max=coef))
data[names(ou$p),]
data <- data[-c(as.numeric(names(ou$p))),]
nrow(data)

# LINEAR REGRESSION MODEL
rownames(data) <- seq_len(nrow(data))
linear_model <- lm(data$VAR_RESP ~ . , data[,-c(which(colnames(data)=="VAR1"),which(colnames(data)=="VAR2") ) ])
summary(linear_model) 
final.linear_model <- stepAIC(linear_model, direction="both")
summary(final.linear_model) 

#MODELO DE REGRESIÓN LOGÍSTICO
rownames(data) <- seq_len(nrow(data))
glm <- glm(data$VAR_RESP ~ . , data[,-c(which(colnames(data)=="VAR1"),which(colnames(data)=="VAR2"))] , family="binomial")
summary(glm) #
final.glm <- stepAIC(glm, direction="both")
summary(final.glm) #
plot(final.glm)

#BORRAR MEMORIA Y CACHÉ SIN AFECTAR WORKSPACE Y VARIABLES CREADAS
gc()

#NORMALIZACIÓN DE MÁRGENES
par(mar=c(5.1,4.1,4.1,2.1))

#ELIMINAR DUPLICADOS
df[!duplicated(df), ]

#DISTANCIA DE COOK
rownames(data) <- seq_len(nrow(data))
cutoff <- 4/((nrow(data)-length(final.linear_model$coefficients)-2)) 
par(mfrow=c(1,1))
#plot(final.linear_model, which=1:6, cook.levels=cutoff)
data <- data[-c(777,4149,6129,9080,9198,8008),]

#ANÁLISIS DE LEVERAGE
data$leverage <- hatvalues(final.linear_model)
nrow(data)
data[data$leverage==1,] #OK. Perfecto
# data <- data[data$leverage<1,]
# nrow(data)
temporal <- names(data) %in% c("leverage")
data <- data[!temporal]

#CAMBIO DE NOMBRES
names(data_set)[names(data_set) == 'Nombre_Anterior'] <- 'Nombre_Nuevo' #Cambiar el nombre

#IMPUTACIÓN DE DATOS FALTANTES
#Técnica MICE
library(mice)
gc()
imp <- mice(data, m=1)
data <- complete(imp)
#Técnica missForest #Aplicable a Big Data
library(missForest)
gc()
imp <- missForest(data[,-c(which(colnames(data)=="CONTROLN"))])
data <- imp$ximp

#COMPROBAR EXISTENCIA DE NA'S
nrow(data[!complete.cases(data),]) #Regresar

#ANÁLISIS DE TRANSFORMACIONES
crPlots(linear_model)

#PREDICCIONES
data$prediccion <- predict(final.lm4,data)
data$error <- abs(data$ViolentCrimesPerPop-data$prediccion)
final <- data[,c("normalized.losses","prediccion","error")]
View(final)

#ORDENAR COLUMNAS DE UN DATA SET EN ORDEN ALFABÉTICO / ORDENAR POR NOMBRE
data <- data[ , order(names(data))]

#ELIMINAR MISSINGS
data <- na.omit(data)
#PROMEDIO EN PORENTAJE DE MISSINGS EN TODO EL DATA SET
mean(is.na(data)) 
#NÚMERO DE MISSINGS POR COLUMNA
sapply(data, function(x) sum(is.na(x)))
#PORCENTAJE DE MISSINGS POR COLUMNA
sapply(data, function(x) ((sum(is.na(x)))*100/length(x)))
#PORCENTAJE DE MISSINGS POR FILA
rowMeans(is.na(data))
#NOMBRES DE COLUMNAS CON MISSINGS MAYOR AL 15%
names(data[,colMeans(is.na(data))<0.7])
#ELIMINAR COLUMNAS CON MISSINGS MAYOR AL 15%
data <- data[, -which(colMeans(is.na(data)) > 0.15)]
#NÚMERO DE MISSINGS
sum(is.na(data))

#REEMPLAZO DE VALORES PARTICULARES POR NA / IMPUTACIÓN DE MISSING /
#REEMPLAZO DE VALORES NULOS, CHARACTER O VACÍOS POR NA
data[data == ""] <- NA

#VER TODOS LOS PAQUETES O LIBRERÍAS INSTALADAS
librerias <- as.data.frame(installed.packages()[,c(1,3:4)])
rownames(librerias) <- NULL
librerias <- librerias[is.na(librerias$Priority),1:2,drop=FALSE]
print(librerias, row.names=FALSE)

#ELIMINAR TODOS LOS OBJETOS EXCEPTO UNO / LIMPIAR ENTORNO DE TRABAJO / LIMPIAR RDATA
#VACIAR ENVIRONMENT / ELIMINAR VARIABLES
rm(list=setdiff(ls(), "x")) #Eliminamos todos excepto x

#ESCRIBIR EN EXCEL
write.xlsx(data, file="Ruta/nombre_del_archivo.xls", row.names=FALSE)

#SIMULACIÓN NUMÉRICA REAL
#Se transforma la matriz que se tenga, sin tener que separar y luego concatenar submatrices
numericas <- sapply(data, is.numeric)
simvarnum <- function(x){
  for(i in 1:nrow(x)) {
    for(j in 1:ncol(x)) {
      x[i,j] <- runif(1,0.85,1.15)*x[i,j]
    }
  }
  return(x)
}
data[,numericas] <- simvarnum(data[,numericas])

#SIMULACIÓN NUMÉRICA REAL
#Se deben escoger las variables
data$nombre_variable <- round(data$nombre_variable,0)

#SIMULACIÓN CATEGÓRICA
v <- data
perc <- 0.2
set.seed(1234)
idx <- sample(1:nrow(data), perc * nrow(data), replace=F)
data_aux <- as.data.frame(data[idx, sapply(data, is.factor)])
data_aux <- sapply(data_aux, function(x) factor(sample(levels(x),nrow(data_aux),replace=T,prob=table(x))))
summary(data_aux)
data[idx,sapply(v, is.factor)] <- data_aux #En caso de tener la simulacion en una matriz

#ELIMINACIÓN DE COLUMNAS CON TODOS LOS VALORES NA O TODOS LOS VALORES MISSING O TODOS LOS VALORES VACÍOS
data <- data[,colSums(is.na(data))<nrow(data)]

# Conversión a factors
factors_a_character <- c(
  "CODIGOPOSTAL",
  "CODIGOPROVINCIA",
  "ID_AUTOLIQUIDACIO",
  "ID_EXPEDIENT",
  "ID_NOTARI",
  "ID_NOTARIA",
  "ID_OFICINACOMPETENT",
  "ID_OFICINAPRESENTACIO",
  "ID_PERS",
  "DATA_ALTAEXPEDIENT",
  "DATA_DOCUMENT",
  "FECHABAJA",
  "MUNICIPIO",
  "NIF",
  "NOMBRE",
  "CODIGOMUNICIPIOINE",
  "CODIGOPROVINCIA",
  "ID_AUTOLIQUIDACIO",
  "ID_EXPEDIENT",
  "ID_NOTARI",
  "ID_NOTARIA",
  "ID_OFICINACOMPETENT",
  "ID_OFICINAPRESENTACIO",
  "ID_PERS"
)
data[,factors_a_character] <- lapply(factors_a_character, function(x) factor(as.character(data[,x])))

#Conversiones de fechas
data$DATA_ALTAEXPEDIENT <- as.Date(data$DATA_ALTAEXPEDIENT, format="%d/%m/%Y")
#mayuscula significa 4 digitos en año -> 11/04/1989
#format es como nos vienen los datos
data$DATA_ALTAEXPEDIENT <- format(data$DATA_ALTAEXPEDIENT,"%Y")
data$DATA_ALTAEXPEDIENT <- as.numeric(as.character(data$DATA_ALTAEXPEDIENT))
fecha_actual <- Sys.Date()
anyo_actual <- as.numeric(as.character(format(fecha_actual,"%Y")))
data$DATA_ALTAEXPEDIENT <- anyo_actual - data$DATA_ALTAEXPEDIENT 

#Conversión de Fecha tipo character 22/06/1989 a numérico AUTOMATICO
#Conversion de formato "%d/%m/%Y", por ej. "22/06/1989" a numerico (1989)
fecha_a_anyo <- c(
  "DATA_ALTAEXPEDIENT",
  "DATA_DOCUMENT",
  "FECHABAJA"
)
data[,fecha_a_anyo] <- lapply(fecha_a_anyo, function(x) as.Date(data[,x], format="%d/%m/%Y"))
data[,fecha_a_anyo] <- lapply(fecha_a_anyo, function(x) format(data[,x],"%Y"))
data[,fecha_a_anyo] <- lapply(fecha_a_anyo, function(x) as.numeric(as.character(data[,x])))
fecha_actual <- Sys.Date()
anyo_actual <- as.numeric(as.character(format(fecha_actual,"%Y")))
data[,fecha_a_anyo] <- lapply(fecha_a_anyo, function(x) anyo_actual-data[,x])

#BORRAR EL WORKSPACE / BORRAR EL ENTORNO DE TRABAJO
rm(list = ls())
#BORRAR EL WORKSPACE / BORRAR EL ENTORNO DE TRABAJO EXCEPTO UN OBJETO
rm(list=setdiff(ls(), "x"))

#ELIMINAR VALORES MISSINGS DE UN STRING
data$VARIABLE <- gsub(" ", "", data$VARIABLE, fixed = TRUE)
