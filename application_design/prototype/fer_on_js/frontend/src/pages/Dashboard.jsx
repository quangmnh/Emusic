import { useNavigate } from "react-router-dom";

function Dashboard() {
  const navigate = useNavigate();
  const onFER = () => {
    navigate("/fer");
  };

  return (
    <>
      <section className="heading">
        <h1>Frontend to use API</h1>
        <p>Main menu</p>
      </section>
      <div className="content">
        <button className="btn" onClick={onFER}>
          Emotion detect
        </button>
      </div>
    </>
  );
}

export default Dashboard;
