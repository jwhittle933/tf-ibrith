package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/image/colornames"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

var labels []string

// HLine ...
func HLine(img *image.RGBA, x1, y, x2 int, col color.Color) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine ...
func VLine(img *image.RGBA, x, y1, y2 int, col color.Color) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect ...
func Rect(img *image.RGBA, x1, y1, x2, y2, width int, col color.Color) {
	for i := 0; i < width; i++ {
		HLine(img, x1, y1+i, x2, col)
		HLine(img, x1, y2+i, x2, col)
		VLine(img, x1+i, y1, y2, col)
		VLine(img, x2+i, y1, y2, col)
	}
}

// MakeTensor ...
func MakeTensor(filename string) (*tf.Tensor, image.Image, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("Error reading file from MakeTensor")
		return nil, nil, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)
	if err != nil {
		fmt.Println("Error decoding image from MakeTensor")
		return nil, nil, err
	}

	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		fmt.Println("Error creating tensor from MakeTensor")
		return nil, nil, err
	}

	graph, input, output, err := DecodeJpegGraph()
	if err != nil {
		fmt.Println("Error decoding jpg from MakeTensor")
		return nil, nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Println("Error starting new session from MakeTensor")
		return nil, nil, err
	}

	defer session.Close()

	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil,
	)
	if err != nil {
		fmt.Println("ERror running session from MakeTensor")
		return nil, nil, err
	}

	return normalized[0], img, nil
}

// DecodeJpegGraph ...
func DecodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	// s := op.NewScope()
	// input = op.Placeholder(s, tf.String)
	// output = op.ExpandDims(s,
	// 	op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
	// 	op.Const(s.SubScope("make_batch"), int32(0)))
	// graph, err = s.Finalize()

	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)

	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()

	return graph, input, output, err
}

// LoadLabels ...
func LoadLabels(labelsFile string) {
	fmt.Println(labelsFile)
	file, err := os.Open(labelsFile)
	if err != nil {
		fmt.Println("Error opening lable file")
		log.Fatal(err)
	}

	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
}

// GetLabel ...
func GetLabel(idx int, probablities []float32, classes []float32) string {
	index := int(classes[idx])
	label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probablities[idx]*100.00)

	return label
}

// AddLabel ...
func AddLabel(img *image.RGBA, x, y, class int, label string) {
	col := colornames.Map[colornames.Names[class]]
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(colornames.Black),
		Face: basicfont.Face7x13,
		Dot:  point,
	}

	Rect(img, x, y-13, (x + len(label)*7), y-6, 7, col)

	d.DrawString(label)
}

//PrintBestLabel ...
func PrintBestLabel(probabilities []float32, labelsFile string) {
	bestIdx := 0
	for i, p := range probabilities {
		if p > probabilities[bestIdx] {
			bestIdx = i
		}
	}
	// Found the best match. Read the string from labelsFile, which
	// contains one line per label.
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
	fmt.Printf("\n\n\nBEST MATCH: (%2.0f%% likely) %s\n\n\n", probabilities[bestIdx]*100.0, labels[bestIdx])
}

func main() {

	now := time.Now()
	year, month, day := now.Date()
	hour, min, sec := now.Clock()

	outputfile := fmt.Sprintf("%v-%v-%v|%v:%v:%v.jpg", month, day, year, hour, min, sec)

	modeldir := "src/saved_models"
	dir := flag.String("dir", "", "Filename in date format in src/saved_models.")
	jpgfile := flag.String("jpg", "", "Path to a JPG image used for input")
	_ = flag.String("out", outputfile, "Path of output JPG for displaying labels. Default is month-day-year|hour:min:sec.jpg")
	labelfile := flag.String("labels", "labels.txt", "Path to file of labels, one per line")
	flag.Parse()

	if *dir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	LoadLabels(*labelfile)

	modelpath := filepath.Join(modeldir, *dir, "frozen_model.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		fmt.Println("Bad model path")
		log.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		fmt.Println("Error importing model")
		log.Fatal(err)
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Println("Error staring session")
		log.Fatal(err)
	}

	tensor, i, err := MakeTensor(*jpgfile)
	if err != nil {
		fmt.Println("Error making tensor")
		log.Fatal(err)
	}

	b := i.Bounds()
	img := image.NewRGBA(b)
	draw.Draw(img, b, i, b.Min, draw.Src)

	// o1 := graph.Operation("detection_boxes")
	// o2 := graph.Operation("detection_scores")
	// o3 := graph.Operation("detection_classes")
	// o4 := graph.Operation("num_detections")

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("Placeholder").Output(0): tensor,
		},
		[]tf.Output{
			// o1.Output(0),
			// o2.Output(0),
			// o3.Output(0),
			// o4.Output(0),
			graph.Operation("final_result").Output(0),
		},
		nil,
	)
	if err != nil {
		fmt.Println("Error running session from main()") // errors out here.
		log.Fatal(err)
	}

	probabilities := output[0].Value().([][]float32)[0]
	PrintBestLabel(probabilities, *labelfile)
	// fmt.Println("output[0].Value()", output[0].Value().([][]float32)[0])
	// fmt.Println(output[0].Value())
	// classes := output[2].Value().([][]float32)[0]
	// boxes := output[0].Value().([][][]float32)[0]

	// curObj := 0

	// for probabilities[curObj] > 0.4 {
	// 	x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
	// 	x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
	// 	y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
	// 	y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

	// 	Rect(img, int(x1), int(y1), int(x2), int(y2), 4, colornames.Map[colornames.Names[int(classes[curObj])]])
	// 	AddLabel(img, int(x1), int(y1), int(classes[curObj]), GetLabel(curObj, probabilities, classes))

	// 	curObj++
	// }

	// outfile, err := os.Create(*outjpg)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// var opt jpeg.Options

	// opt.Quality = 80

	// err = jpeg.Encode(outfile, img, &opt)
	// if err !=nil {
	// 	log.Fatal(err)
	// }

}
