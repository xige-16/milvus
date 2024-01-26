package querynodev2

import (
	"errors"
	"fmt"
	"github.com/containerd/cgroups"
	_ "golang.org/x/exp/rand"
	_ "net/http/pprof"
	_ "path"
	_ "time"
)

const (
	filePath  = "/tmp/data"
	batchSize = 500 * 1024 * 1024
	fileSize  = 5 * 1024 * 1024 * 1024
)

func PrintActiveInactive() {
	control, err := cgroups.Load(cgroups.V1, cgroups.RootPath)
	if err != nil {
		panic(err)
	}
	stats, err := control.Stat(cgroups.IgnoreNotExist)
	if err != nil {
		panic(err)
	}
	if stats.Memory == nil || stats.Memory.Usage == nil {
		panic(errors.New("cannot find memory usage info from cGroups"))
	}
	// 1. usage
	// ref: <https://github.com/docker/cli/blob/e57b5f78de635e6e2b688686d10b830c4747c4dc/cli/command/container/stats_helpers.go#L239>
	//usage := stats.Memory.Usage.Usage
	//if inactiveFile < usage {
	//	usage = usage - stats.Memory.TotalInactiveFile
	//}
	//fmt.Printf("total:%.2fMB\n", float64(stats.Memory.Usage.Limit)/1024/1024)
	//fmt.Printf("used:%.2fMB\n", float64(usage)/1024/1024)
	fmt.Printf("%.2f ", float64(stats.Memory.TotalInactiveFile)/1024/1024)
	fmt.Printf("%.2f ", float64(stats.Memory.TotalActiveFile)/1024/1024)
}

func PrintMeminfo() {
	control, err := cgroups.Load(cgroups.V1, cgroups.RootPath)
	if err != nil {
		panic(err)
	}
	stats, err := control.Stat(cgroups.IgnoreNotExist)
	if err != nil {
		panic(err)
	}
	if stats.Memory == nil || stats.Memory.Usage == nil {
		panic(errors.New("cannot find memory usage info from cGroups"))
	}
	// 1. usage
	// ref: <https://github.com/docker/cli/blob/e57b5f78de635e6e2b688686d10b830c4747c4dc/cli/command/container/stats_helpers.go#L239>
	usage := stats.Memory.Usage.Usage
	if stats.Memory.TotalInactiveFile < usage {
		usage = usage - stats.Memory.TotalInactiveFile - stats.Memory.TotalActiveFile
	}
	fmt.Printf("----------------\n")
	fmt.Printf("total:%.4fGB\n", float64(stats.Memory.Usage.Limit)/1024/1024/1024)
	fmt.Printf("used:%.4GGB\n", float64(usage)/1024/1024/1024)
	fmt.Printf("usedWithCache:%.4fGB\n", float64(stats.Memory.Usage.Usage)/1024/1024/1024)
	fmt.Printf("inactive(file):%.2fMB\n", float64(stats.Memory.TotalInactiveFile)/1024/1024)
	fmt.Printf("active(file):%.2fMB\n", float64(stats.Memory.TotalActiveFile)/1024/1024)
	//fmt.Printf("inactive(anon):%.2fMB\n", float64(stats.Memory.TotalInactiveAnon)/1024/1024)
	//fmt.Printf("active(anon):%.2fMB\n", float64(stats.Memory.TotalActiveAnon)/1024/1024)
	//fmt.Printf("cached:%.2fMB\n", float64(stats.Memory.TotalCache)/1024/1024)
	//fmt.Printf("mapped:%.2fMB\n", float64(stats.Memory.TotalMappedFile)/1024/1024)
	fmt.Printf("----------------\n")
}
