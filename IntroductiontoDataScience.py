#importando  pandas
# Import pandas
import pandas as pd

#=======================================
#cargar csv e imprimir esta información
# Load the 'ransom.csv' into a DataFrame
r = pd.read_csv('ransom.csv')

# Display DataFrame
print(r)

#=======================================
#cargamos un csv e imprimimos los 5 primero registros e imprimos el info 
#para ver columnas y  filas 
# Import pandas under the alias pd
import pandas as pd
# Load the CSV "credit_records.csv"
credit_records = pd.read_csv('credit_records.csv')

# Display the first five rows of credit_records using the .head() method
print(credit_records.head())

#Use .info() to inspect the DataFrame credit_records
print(credit_records.info())



#======================================================
#imprimos de la tabla credit_records la columna items y muestra los resultados

# Select the column item from credit_records
# Use brackets and string notation
items = credit_records['item']

# Display the results
print(items)

#=====================================================================
#seleccionamos de table credit_records, la columna locationsy la imprimimos
# One or more lines of code contain errors.
# Fix the errors so that the code runs.

# Select the location column in credit_records
location = credit_records['location']

# Select the item column in credit_records
items = credit_records.item

# Display results
print(location)

#=====================================================================
#imprimimos  la información de la tabla mpr 
# Use info() to inspect mpr
print(mpr.info())

#=====================================================================
#imprimimos  en columna  height_inches mayor a  70

# Is height_inches greater than 70 inches?
print(height_inches > 70)

#=====================================================================
#guardamos variable de la data mpr , se leccionamos mpr.age mayor a 2 
#guardamos  en variable  de la data mpr, seleccionamos mpr.status  == "desaparecido"
#guardamos  en variable  de la data mpr, seleccionamos mpr.breed  == "poodle"
#seleccionamos  de la tabla credit_records, seleccionamos credit_records.location == "pet paradise"

# Select the dogs where Age is greater than 2
greater_than_2 = mpr[mpr.Age > 2]
print(greater_than_2)

# Select the dogs whose Status is equal to Still Missing
still_missing = mpr[mpr.Status == 'Still Missing']
print(still_missing)

# Select all dogs whose Dog Breed is not equal to Poodle
not_poodle = mpr[mpr['Dog Breed'] != 'Poodle']
print(not_poodle)

# Select purchases from 'Pet Paradise'
purchase = credit_records[credit_records.location == 'Pet Paradise']

# Display
print(purchase)


#=====================================================================
#plot, lo usamos para hacer gráficas, seleccionamos en x de la  tabla  deshaun la columna  day_of_week y en el eje de la y sera deshaun.hous_worked
#creara  una gráfica  o plot 

# From matplotlib, import pyplot under the alias plt
from matplotlib import pyplot as plt

# Plot Officer Deshaun's hours_worked vs. day_of_week
plt.plot(deshaun.day_of_week, deshaun.hours_worked)

# Plot Officer Aditya's hours_worked vs. day_of_week
plt.plot(aditya.day_of_week, aditya.hours_worked)

# Plot Officer Mengfei's hours_worked vs. day_of_week
plt.plot(mengfei.day_of_week, mengfei.hours_worked)

# Display all three line plots
plt.show()


#=====================================================================
#plot, lo usamos para hacer gráficas, seleccionamos en x de la  tabla  deshaun la columna  day_of_week y en el eje de la y sera deshaun.hous_worked
#creara  una gráfica  o plot, aqui agregamos el label para indicar que linea pertenece a que datos

# Add a label to Deshaun's plot
plt.plot(deshaun.day_of_week, deshaun.hours_worked, label="Deshaun")

# Officer Aditya
plt.plot(aditya.day_of_week, aditya.hours_worked)

# Officer Mengfei
plt.plot(mengfei.day_of_week, mengfei.hours_worked)

# Display plot
plt.show()

#=====================================================================
#plot, lo usamos para hacer gráficas, seleccionamos en x de la  tabla  deshaun la columna  day_of_week y en el eje de la y sera deshaun.hous_worked
#creara  una gráfica  o plot, aqui agregamos el label para indicar que linea pertenece a que datos
#agregamos title en la  gráfica
# en el eje y ponemos que ponemos  el label de  "day of week"


# Lines
plt.plot(deshaun.day_of_week, deshaun.hours_worked, label='Deshaun')
plt.plot(aditya.day_of_week, aditya.hours_worked, label='Aditya')
plt.plot(mengfei.day_of_week, mengfei.hours_worked, label='Mengfei')

# Add a title
plt.title("Hours worked officer")

# Add y-axis label
plt.ylabel("day of week")

# Legend
plt.legend()
# Display plot
plt.show()


#=====================================================================
#gráficamos  de la tabla de seis meses, los meses en el eje x 
#gráficamos de la tabla  de seis meses, las horas trabajadas
#agregamos texto en el sector 2.5, 80  un  texto de gráfica

# Create plot
plt.plot(six_months.month, six_months.hours_worked)

# Add annotation "Missing June data" at (2.5, 80)
plt.text(2.5, 80, "Missing June data")

# Display graph
plt.show()


#=====================================================================
#seleccionamos del data  la columnas year  y  phoenix police dept
#ponemos nuetro label de  phoenix y el color que es el color de linea 
#en la siguiente linea  se gráfica del  dataframe data, la columna "year"  y "los angeles police dept"
#cambiamos  la linea de estilo  con :
#en la siguiente ponemos  la marca en cada  parte de la gráfica 
# Change the color of Phoenix to `"DarkCyan"`
plt.plot(data["Year"], data["Phoenix Police Dept"],
         label="Phoenix", color="DarkCyan")

# Make the Los Angeles line dotted
plt.plot(data["Year"], data["Los Angeles Police Dept"],
         label="Los Angeles", linestyle=":")

# Add square markers to Philedelphia
plt.plot(data["Year"], data["Philadelphia Police Dept"],
         label="Philadelphia", marker="s")

# Add a legend
plt.legend()

# Display the plot
plt.show()


#=====================================================================
#usamos, el cambio de  style de plt 


# Change the style to fivethirtyeight
plt.style.use('fivethirtyeight')

# Plot lines
plt.plot(data["Year"], data["Phoenix Police Dept"], label="Phoenix")
plt.plot(data["Year"], data["Los Angeles Police Dept"], label="Los Angeles")
plt.plot(data["Year"], data["Philadelphia Police Dept"], label="Philadelphia")

# Add a legend
plt.legend()

# Display the plot
plt.show()


#=====================================================================
#gráficamos , colocamos labels, cambiamos las lineas de estilo y el color de  la linea



# x should be ransom.letter and y should be ransom.frequency
plt.plot(ransom.letter, ransom.frequency,
         # Label should be "Ransom"
         label="Ransom",
         # Plot the ransom letter as a dotted gray line
         linestyle=':', color='gray')

# Display the plot
plt.show()


# Plot each line
plt.plot(ransom.letter, ransom.frequency,
         label='Ransom', linestyle=':', color='gray')

# X-values should be suspect1.letter
# Y-values should be suspect1.frequency
# Label should be "Fred Frequentist"
plt.plot(suspect1.letter, suspect1.frequency, label="Fred Frequentist")

# Display the plot
plt.show()

# Plot each line
plt.plot(ransom.letter, ransom.frequency,
         label='Ransom', linestyle=':', color='gray')
plt.plot(suspect1.letter, suspect1.frequency,
         label='Fred Frequentist')

# X-values should be suspect2.letter
# Y-values should be suspect2.frequency
# Label should be "Gertrude Cox"
plt.plot(suspect2.letter, suspect2.frequency, label="Gertrude Cox")

# Display plot
plt.show()


# Plot each line
plt.plot(ransom.letter, ransom.frequency,
         label='Ransom', linestyle=':', color='gray')
plt.plot(suspect1.letter, suspect1.frequency, label='Fred Frequentist')
plt.plot(suspect2.letter, suspect2.frequency, label='Gertrude Cox')

# Add x- and y-labels
plt.xlabel("Letter")
plt.ylabel("Frequency")

# Add a legend
plt.legend()

# Display plot
plt.show()

# Explore the data
print(cellphone.head())

# Create a scatter plot of the data from the DataFrame cellphone
plt.scatter(cellphone.x, cellphone.y)

# Add labels
plt.ylabel('Latitude')
plt.xlabel('Longitude')

# Display the plot
plt.show()


# Change the marker color to red
plt.scatter(cellphone.x, cellphone.y,
            color="red")

# Add labels
plt.ylabel('Latitude')
plt.xlabel('Longitude')

# Display the plot
plt.show()

# Display the DataFrame hours using print
print(hours)

# Create a bar plot from the DataFrame hours
plt.bar(hours.officer, hours.avg_hours_worked,
        # Add error bars
        yerr=hours.std_hours_worked)

# Display the plot
plt.show()

#=====================================================================
#hacemos un histograma 

# Create a histogram of the column weight
# from the DataFrame puppies
plt.hist(puppies.weight)

# Add labels
plt.xlabel('Puppy Weight (lbs)')
plt.ylabel('Number of Puppies')

# Display
plt.show()

#=====================================================================
#hacemos un histograma, ponemos el bins y los rangos 

# Create a histogram
plt.hist(gravel.radius,
         bins=40,
         range=(2, 8),
         density=True)

# Label plot
plt.xlabel('Gravel Radius (mm)')
plt.ylabel('Frequency')
plt.title('Sample from Shoeprint')

# Display histogram
plt.show()
