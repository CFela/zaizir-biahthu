const { createContext } = require("react");
const { io } = require("socket.io-client");

const SocketContext = createContext(null);

const useSocket = () => {
  return SocketContext.socket;
};

const SocketProvider = (props) => {
  const socket = io("localhost:8000");

  SocketContext.socket = socket;

  return props.children;
};

module.exports = { useSocket, SocketProvider };
