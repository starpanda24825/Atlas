from core.daemon import AtlasDaemon

daemon = AtlasDaemon()
try:
    daemon.run()
except KeyboardInterrupt:
    daemon.shutdown()
