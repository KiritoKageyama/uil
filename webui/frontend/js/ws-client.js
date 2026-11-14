// One WebSocket per active run. onSnapshot/onFinished/onError are set
// by app.js; this file only owns the connection lifecycle.
const WsClient = {
  socket: null,
  onSnapshot: null,
  onFinished: null,
  onError: null,

  connect(runId) {
    this.disconnect();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
    this.socket.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      const t = msg.type || msg.event;
      if ((t === "stage" || t === "started") && this.onSnapshot) this.onSnapshot(msg);
      else if (t === "finished" && this.onFinished) this.onFinished(msg);
      else if (t === "error" && this.onError) this.onError(msg);
    };
  },

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  },
};
