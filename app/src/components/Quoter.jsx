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
import Modal from "./Modal";

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

  const [acceptance, setAcceptance] = useState(false);
  const [age, setAge] = useState(0);
  const [origin, setOrigin] = useState(null);
  const [destination, setDestination] = useState(null);
  const [budget, setBudget] = useState(50);
  const [isVIP, setIsVIP] = useState(false);
  const [isCryoSleep, setIsCryoSleep] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  

  const handleQuoteForAPI = async (event) => {
    try {
      console.log("Sending info");
      const response = await fetch("http://52.45.99.191:8080/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          Budget: isCryoSleep ? 0 : budget,
          Age: age,
          VIP_True: isVIP ? 1.0 : 0.0,
          CryoSleep_True: isCryoSleep ? 1.0 : 0.0,
          HomePlanet_Mars: origin === 1 ? 1.0 : 0.0,
          "Destination_PSO J318.5-22": destination === 2 ? 1.0 : 0.0,
          "Destination_TRAPPIST-1e": destination === 0 ? 1.0 : 0.0,
        }),
      });

      const requestBody = {
        Budget: isCryoSleep ? 0 : budget,
        Age: age,
        VIP_True: isVIP ? 1.0 : 0.0,
        CryoSleep_True: isCryoSleep ? 1.0 : 0.0,
        HomePlanet_Mars: origin === 1 ? 1.0 : 0.0,
        "Destination_PSO J318.5-22": destination === 2 ? 1.0 : 0.0,
        "Destination_TRAPPIST-1e": destination === 0 ? 1.0 : 0.0,
      };
      console.log("Request Body:", requestBody);

      if (response.ok) {
        const result = await response.json();
        console.log(result);
        console.log(result.Prediction);
        setAcceptance(result.Prediction);
        setIsModalOpen(true);
      } else {
        console.error("Error en la respuesta:", response.statusText);
      }
    } catch (err) {
      console.log("Founded error: ", err);
    }
  };

  const handlePlanetClick = (type, planetIndex) => {
    if (type === "origin") setOrigin(planetIndex);
    if (type === "destination") setDestination(planetIndex);
    console.log("Planet origin: " + origin);
    console.log("Planet destination: " + destination);
  };

  const handleArrowClick = () => {
    navigate("/");
  };

  const handleVIPChange = () => setIsVIP(!isVIP);

  const handleCryoSleepChange = () => {
    setIsCryoSleep(!isCryoSleep);
    if (!isCryoSleep) setBudget(0);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  return (
    <div className="quoter-page">
      <img
        src={backgroundQuoterGif}
        alt="Log In Gif"
        className="quoter-background"
      />
      <div className="quoter-header" onClick={handleArrowClick}>
        <FaArrowLeftLong className="arrow" />
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
              <div key={index} className="planet-container">
                <img
                  src={planet.imgs}
                  alt={planet.name}
                  className={`planet ${origin === index ? "selected" : ""}`}
                  onClick={() => handlePlanetClick("origin", index)}
                />
                <div className="planet-name">{planet.name}</div>{" "}
              </div>
            ))}
          </div>
        </div>

        <div className="destination-section">
          <h3>Destination</h3>
          <div className="planet-options">
            {destinationPlanets.map((planet, index) => (
              <div key={index} className="planet-container">
                <img
                  src={planet.imgs}
                  alt={planet.name}
                  className={`planet ${
                    destination === index ? "selected" : ""
                  }`}
                  onClick={() => handlePlanetClick("destination", index)}
                />
                <div className="planet-name">{planet.name}</div>{" "}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="cryo-section">
        <label>
          <input
            type="checkbox"
            checked={isCryoSleep}
            onChange={handleCryoSleepChange}
          />{" "}
          Are you gonna travel in Cryo Sleep?
        </label>
      </div>

      <div className="budget-section">
        <label>What's your budget?</label>
        <input
          type="range"
          min="0"
          max="6000"
          value={isCryoSleep ? 0 : budget}
          onChange={(e) =>
            setBudget(e.target.value)
          }
          className="budget-slider"
          disabled={isCryoSleep}
        />
        <div>Budget: ${isCryoSleep ? 0 : budget}</div>
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

      {isModalOpen && (
        <Modal acceptance={acceptance} onClose={handleCloseModal} destin={destination}/>
      )}
    </div>
  );
};

export default Quoter;