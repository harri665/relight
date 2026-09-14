# Harrison Default Web Template

A full-stack web application template with React frontend and Express backend.

## Project Structure

```
HarrisonDefaultWeb/
├── client/          # React frontend (Vite + React 19)
├── server/          # Express backend
└── package.json     # Root package with concurrent scripts
```

## Tech Stack

### Frontend (Client)
- **React 19** - UI library
- **React Router DOM** - Client-side routing
- **Vite** - Build tool and dev server
- **Tailwind CSS 4** - Utility-first CSS framework
- **ESLint** - Code linting

### Backend (Server)
- **Express 5** - Web framework
- **CORS** - Cross-origin resource sharing
- **Nodemon** - Development auto-restart

## Getting Started

### Prerequisites
- Node.js (v16 or higher recommended)
- npm

### Installation

1. Clone the repository:
```bash
git clone https://github.com/harri665/HarrisonWebDefault.git
cd HarrisonDefaultWeb
```

2. Install dependencies for all packages:
```bash
npm install
npm run install:all
```

### Development

Start both client and server concurrently:
```bash
npm run dev
```

Or run them individually:
```bash
npm run client    # Start frontend only
npm run server    # Start backend only
```

The client will typically run on `http://localhost:5173` and the server on `http://localhost:3001` (or ports configured in your setup).

To change port for client or servercreate a .env in the respective folder with: 
```env
PORT: #port number here 
```


## Available Scripts

### Root Level
- `npm run dev` - Run both client and server concurrently
- `npm run client` - Run client development server only
- `npm run server` - Run server development server only
- `npm run install:all` - Install dependencies in both client and server

### Client (`cd client`)
- `npm run dev` - Start Vite dev server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Server (`cd server`)
- `npm run dev` - Start server with nodemon (auto-restart)
- `npm start` - Start server in production mode


