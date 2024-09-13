import "../styles/quoter.css";
import { useNavigate } from "react-router-dom";
import React, { useState } from "react";
import { FaArrowLeftLong } from "react-icons/fa6";
import logoWhite from "../assets/logoWhite.png";
import backgroundQuoterGif from "../assets/backIdea3.gif";
import d1 from "../assets/d1.png";
import d2 from "../assets/d2.png";
import d3 from "../assets/d3.png";
import mars from "../assets/mars.png";
import earth from "../assets/earth.png";

const homePlanets = [
  { name: "Earth", color: "#50D4F2", imgs: earth },
  { name: "Mars", color: "#F5913A", imgs: mars },
];
const destinationPlanets = [
  { name: "TRAPPIST-1e", color: "#F7E080", imgs: d1 },
  { name: "55 Cancri e", color: "#db6bb0", imgs: d2 },
  { name: "PSO J318.5-22", color: "#B4E351", imgs: d3 },
];

const Quoter = (props) => {
  const navigate = useNavigate();

  const [age, setAge] = useState("");
  const [origin, setOrigin] = useState(null);
  const [destination, setDestination] = useState(null);
  const [budget, setBudget] = useState(50);
  const [isVIP, setIsVIP] = useState(false);

  const handleQuoteForAPI = async (event) => {
    try{
        console.log('Sending info')
        const response = await fetch('http://52.45.99.191:8080/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                RoomService: 5,
                FoodCourt: 5,
                ShoppingMall: 5,
                Spa: 5,
                VRDeck: 5,
                Age: age,
                VIP_True: 1.0,
                CryoSleep_True: 0.0,
                HomePlanet_Mars: 1.0,
                'Destination_PSO J318.5-22': 0.0,
                'Destination_TRAPPIST-1e': 1.0
            })
        });
        if(!response.ok){
            throw new Error('Failed to fetch');
        }
        console.log(response.json());
    }
    catch(err){
        console.log('Founded error: ',err);
    }
  }

  const handlePlanetClick = (type, planetIndex) => {
    if (type === "origin") setOrigin(planetIndex);
    if (type === "destination") setDestination(planetIndex);
  };

  const handleArrowClick = () => {
    navigate("/");
  };

  const handleVIPChange = () => setIsVIP(!isVIP);

  const handleQuote = () => {
    alert(
      `Age: ${age}\nOrigin: Planet ${origin + 1}\nDestination: Planet ${
        destination + 1
      }\nBudget: ${budget}\nVIP: ${isVIP}`
    );
  };

  return (
    <div className="quoter-page">
      <img
        src={backgroundQuoterGif}
        alt="Log In Gif"
        className="quoter-background"
      />
      <div className="quoter-header" onClick={handleArrowClick}>
        <FaArrowLeftLong class="arrow" />
      </div>
      <div>
        <p className="quoter-title">Stellar Exodus Quoter</p>
        <img src={logoWhite} alt="Logo" className="mini-logo" />
      </div>

      <div className="form-section">
        <label>Enter your age</label>
        <input
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          className="input-field"
        />
      </div>

      <div className="planet-section">
        <div className="origin-section">
          <h3>Origin</h3>
          <div className="planet-options">
            {homePlanets.map((planet, index) => (
              <img
                src={planet.imgs}
                alt={planet.name}
                key={index}
                className={`planet ${origin === index ? "selected" : ""}`}
                onClick={() => handlePlanetClick("origin", index)}
              />
            ))}
          </div>
        </div>

        <div className="destination-section">
          <h3>Destination</h3>
          <div className="planet-options">
            {destinationPlanets.map((planet, index) => (
              <img
                src={planet.imgs}
                alt={planet.name}
                key={index}
                className={`planet ${destination === index ? "selected" : ""}`}
                onClick={() => handlePlanetClick("destination", index)}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="budget-section">
        <label>What's your budget?</label>
        <input
          type="range"
          min="0"
          max="100"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
          className="budget-slider"
        />
        <div>Budget: {budget}</div>
      </div>

      <div className="vip-section">
        <label>
          <input type="checkbox" checked={isVIP} onChange={handleVIPChange} />{" "}
          Are you a VIP member of Stellar Exodus?
        </label>
      </div>

      <button onClick={handleQuoteForAPI} className="quote-button">
        Quote
      </button>
    </div>
  );
};

export default Quoter;
