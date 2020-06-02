sbt-shell:
	sbt -J-XX:MaxMetaspaceSize=2000m
build:
	sbt assembly
submit:
	spark-submit --class raywu60kg.kaggles.titanic.TitanicLogisticRegression --master local[*] target/scala-2.11/spark-scala-kaggles-assembly-0.1.jar data/titanic/train.csv data/titanic/test.csv /tmp/submit
