import './App.css';
import MainPage from './views/MainPage';

function App() {

  // const myStyle={
  //   backgroundImage: "url('https://wallpaperaccess.com/full/4554.jpg')",
  //   height:'100vh',
  //   backgroundSize: 'cover',
  //   backgroundRepeat: 'no-repeat',
  // };

  return (
    <div className="App">
      <img className="bg" src="https://wallpaperaccess.com/full//4554.jpg" alt="backgroundImage" />
      <div className="content">
        <MainPage />
      </div>
    </div>
  );
}

export default App;
