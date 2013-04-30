package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"github.com/jbrukh/bayesian"
	"io"
	"os"
)

const (
	C0 bayesian.Class = "0"
	C1 bayesian.Class = "1"
	C2 bayesian.Class = "2"
	C3 bayesian.Class = "3"
	C4 bayesian.Class = "4"
	C5 bayesian.Class = "5"
	C6 bayesian.Class = "6"
	C7 bayesian.Class = "7"
	C8 bayesian.Class = "8"
	C9 bayesian.Class = "9"
)

func main() {
	flag.Parse()
	if flag.NArg() < 2 {
		fmt.Println("usage: train TRAINFILE TESTFILE")
		os.Exit(1)
	}

	// Retrieve filenames.
	trainFilename := flag.Arg(0)
	testFilename := flag.Arg(1)
	fmt.Printf("FILES: %v, %v\n\n", trainFilename, testFilename)

	// Open training file.
	file, err := os.Open(trainFilename)
	if err != nil {
		fmt.Printf("File not found: %v\n", trainFilename)
		os.Exit(1)
	}
	defer file.Close()

	// Create classifier.
	classifier := bayesian.NewClassifier(C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)

	// Parse training file.
	csvReader := csv.NewReader(bufio.NewReader(file))
	index := 0
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error in CSV!")
			os.Exit(1)
		}

		if index > 0 {
			// Normalize data.
			label := bayesian.Class(record[0])
			data := []string{}
			for i, value := range record[1:] {
				data = append(data, fmt.Sprintf("%d:%s", i, value))
			}

			// Classify data.
			classifier.Learn(data, label)
		}
		index++
	}
}
