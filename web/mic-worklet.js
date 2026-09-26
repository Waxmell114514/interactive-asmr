// 麦克风采集：把 AudioContext 采样率的输入降到 16kHz，每 20ms 打包成 Int16 PCM 发给主线程。
class MicCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const target = options.processorOptions?.targetRate || 16000;
    this.ratio = sampleRate / target;
    this.frame = Math.round(target * 0.02);
    this.out = new Int16Array(this.frame);
    this.n = 0;
    this.acc = 0;
    this.count = 0;
    this.phase = 0;
    this.peak = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // 盒式滤波 + 抽取：对每个输出样本覆盖的输入样本取平均，顺便抗混叠
      this.acc += ch[i];
      this.count++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.count));
        this.acc = 0;
        this.count = 0;
        this.peak = Math.max(this.peak, Math.abs(v));
        this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
        if (this.n === this.frame) {
          this.port.postMessage({ pcm: this.out.buffer, peak: this.peak }, [this.out.buffer]);
          this.out = new Int16Array(this.frame);
          this.n = 0;
          this.peak = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
